# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

import functools
import logging
from textwrap import dedent
from typing import Any, Literal, Optional, TYPE_CHECKING

from kbel.core.mention import Mention
from kbel.core.results import DisambiguationResult

from ..abc import Candidate, Disambiguator
from .constants import EL_DEFAULT_EXAMPLES, EL_DEFAULT_PROMPT
from .parsers import CommaSeparatedListOutputParserSet
from .utils import build_model

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

LOG = logging.getLogger(__name__)


class LLM_Disambiguator(Disambiguator, strategy_name='llm'):
    """Disambiguator that leverages a large language model (LLM) for entity disambiguation.

    This disambiguator uses a chat-based LLM to select the most appropriate candidate
    entity for a given label. The disambiguation is performed by constructing a prompt
    with candidate information and optional textual context.

    Example:
        >>> disamb = LLM_Disambiguator(
        ...     strategy_name='llm',
        ...     model_name='gpt-4',
        ...     model_provider='openai',
        ...     model_endpoint='https://api.openai.com/v1',
        ...     model_apikey='MY_API_KEY'
        ... )
        >>> candidates = [
        ...     {"label": "Python", "description": "Programming language", "iri": "https://www.wikidata.org/wiki/Q28865"},
        ...     {"label": "Python regius", "description": "species of reptile", "iri": "https://www.wikidata.org/wiki/Q464424"}
        ... ]
        >>> disamb._disambiguate("Python", candidates, sentence="Python is used in coding")
        [('Python', 'Programming language', 'https://www.wikidata.org/wiki/Q28865')]
    """

    _model: "BaseChatModel"

    def __init__(
        self,
        strategy_name: str,
        model: Optional["BaseChatModel"] = None,
        model_name: Optional[str] = None,
        model_provider: Optional[Literal['ibm', 'openai', 'ollama']] = None,
        model_params: dict[str, Any] = {},
        model_apikey: Optional[str] = None,
        model_endpoint: Optional[str] = None,
    ):
        """Initializes the LLM_Disambiguator.

        Either an existing model can be provided, or it will be built from
        the specified model parameters.

        Args:
            strategy_name (str): Name of this strategy.
            model (Optional[BaseChatModel]): Pre-initialized LLM model.
            model_name (Optional[str]): Name of the model to load if `model` is None.
            model_provider (Optional[Literal['ibm', 'openai', 'ollama']]): LLM provider.
            model_params (dict[str, Any]): Additional parameters for building the model.
            model_apikey (Optional[str]): API key for model access.
            model_endpoint (Optional[str]): Endpoint for the LLM.
            *args: Additional positional arguments for the base Disambiguator.
            **kwargs: Additional keyword arguments for the base Disambiguator.
        """
        try:
            from langchain_core.language_models import BaseChatModel
        except ModuleNotFoundError as e:
            raise ImportError(
                "LLM support is not available because the optional dependencies are not installed.\n\n"
                "Install them with:\n\n"
                "    pip install 'kbel[llm]'"
            ) from e
        
        assert strategy_name == self.strategy_name
        super().__init__()

        if model:
            self._model = model
        else:
            assert model_name and model_provider and model_endpoint and model_apikey
            _model = build_model(
                model_name=model_name,
                provider=model_provider,
                endpoint=model_endpoint,
                apikey=model_apikey,
                **model_params)

            assert _model
            self._model = _model

    @property
    def model(self) -> "BaseChatModel":
        return self._model

    def _disambiguate(
        self,
        mention: Mention,
        candidates: list[Candidate],
        limit=100,
        *args,
        **kwargs,
    ) -> list[DisambiguationResult]:
        """Disambiguates a mention using the LLM.

        Args:
            mention (Mention): The mention to disambiguate.
            candidates (list[Candidate]): List of candidate entities.
            *args: Additional positional arguments.
            **kwargs: Keyword arguments, including:
                - sentence (str): Sentence containing the term.
                - textual_context (str, optional): Optional context to guide the LLM.

        Returns:
            list[DisambiguationResult]: List of disambiguation results.
        """

        chain, inputs = self._prepare(mention, candidates)
        try:
            entity_ids = chain.invoke(inputs)
        except Exception as e:
            LOG.warning(f'Exceptions occured while disambiguating the term `{mention.term}`: {e}')
            raise
        return self._results(mention, candidates, entity_ids, limit)

    async def _adisambiguate(
        self,
        mention: Mention,
        candidates: list[Candidate],
        limit=100,
        *args,
        **kwargs,
    ) -> list[DisambiguationResult]:
        """Async version of `_disambiguate`, using the model's native `ainvoke`."""
        chain, inputs = self._prepare(mention, candidates)
        try:
            entity_ids = await chain.ainvoke(inputs)
        except Exception as e:
            LOG.warning(f'Exceptions occured while disambiguating the term `{mention.term}`: {e}')
            raise
        return self._results(mention, candidates, entity_ids, limit)

    def _prepare(
            self,
            mention: Mention,
            candidates: list[Candidate],
    ) -> tuple[Any, dict[str, Any]]:
        """Builds the LLM chain and its inputs for a mention.

        Returns:
            (chain, inputs): A LangChain runnable (prompt | model | parser)
            and the variables to invoke it with.
        """
        assert mention.term, 'Mention term can not be undefined.'
        assert mention.text, 'Mention text can not be undefined.'
        assert candidates, f'No candidates to disambiguate the label `{mention.term}`'

        c_prompt = ''.join(
            f'        ID: {c.id}\n        Term: {c.label.strip()}\n'
            + (f'        Description: {c.description.strip()}\n' if c.description else '')
            + '\n'
            for c in candidates)
        context = mention.context if mention.context and mention.context != mention.text else None
        return self._chain(bool(context)), {
            'context': context,
            'sentence': mention.text,
            'term': mention.term,
            'candidates': c_prompt,
        }

    @functools.cache
    def _chain(self, with_context: bool) -> Any:
        """The prompt | model | parser chain, built once per prompt shape."""
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.runnables import RunnableLambda

        context_line = '\n    Context: "{context}"' if with_context else ''
        u_template = dedent(f"""Now follow the format strictly.\n
Input:
    Sentence: "{{sentence}}"{context_line}
    Term: "{{term}}"

    Candidates:
{{candidates}}
Output:""")
        prompt_template = ChatPromptTemplate.from_messages([
            ('system', EL_DEFAULT_PROMPT + '\n\nExamples:\n' + EL_DEFAULT_EXAMPLES),
            ('human', u_template),
        ])
        debug = RunnableLambda(lambda entry: (LOG.debug(entry), entry)[1])
        return (prompt_template | debug | self.model | debug
                | CommaSeparatedListOutputParserSet() | debug)
    @staticmethod
    def _results(
            mention: Mention,
            candidates: list[Candidate],
            entity_ids: list[str],
            limit: Optional[int] = None,
    ) -> list[DisambiguationResult]:
        """Maps the IDs chosen by the LLM back to the candidates.

        Raises:
            ValueError: If the LLM did not choose any of the candidates.
        """
        by_id = {c.id: c for c in candidates if c.id}
        disamb_entities = [
            DisambiguationResult(candidate=by_id[i])
            for i in entity_ids if i in by_id
        ]
        if disamb_entities:
            return disamb_entities[:limit] if limit else disamb_entities
        LOG.warning('term=%r raw_ids=%r candidate_ids=%r',
                    mention.term, entity_ids, list(by_id)[:5])
        raise ValueError(f'Could not disambiguate the term `{mention.term}` among the candidates.')
