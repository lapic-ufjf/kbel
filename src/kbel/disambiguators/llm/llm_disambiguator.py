# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

import logging
from textwrap import dedent
from typing import Any, Literal, Optional, Tuple, TYPE_CHECKING

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

        assert mention.term, 'Mention term can not be undefined.'

        assert mention.text, 'Mention text can not be undefined.'

        return self.__llm_entity_disambiguation(
            mention,
            candidates,
            limit=limit)

    def __llm_entity_disambiguation(
            self,
            mention: Mention,
            candidates: list[Candidate],
            limit: Optional[int] = None
    ) -> list[DisambiguationResult]:
        """Internal method that executes the LLM-based disambiguation.

        Constructs a prompt with candidates and optional context, invokes the LLM,
        parses the output, and returns the selected candidate(s).

        Args:
            mention (Mention): The mention to disambiguate.
            candidates (list[Candidate]): Candidate entities.
            limit (Optional[int]): Maximum number of results to return.

        Returns:
            list[DisambiguationResult]: List of disambiguation results.

        Raises:
            ValueError: If the LLM cannot disambiguate the term among the candidates.
        """
        assert candidates and len(candidates) > 0, f'No candidates to disambiguate the label `{mention.term}`'
        try:
            c_prompt = ''
            for candidate in candidates:
                c_prompt += f'        ID: {candidate.id}'

                c_label = candidate.label
                if mention.term:
                    c_label = c_label.strip()
                    c_prompt += f'\n        Term: {c_label}'
                description = candidate.description
                if description:
                    description = description.strip()
                    c_prompt += f'\n        Description: {description}'  # noqa E501
                c_prompt += '\n\n'
            s_template = EL_DEFAULT_PROMPT + '\n\nExamples:\n' + EL_DEFAULT_EXAMPLES
            context_template = 'Context: {context}' if mention.context else ''
            u_template = dedent(f"""Now follow the format strictly.\n
Input:
    Sentence: "{{sentence}}"
    Term: "{{term}}"
    {context_template}

    Candidates:
{{candidates}}
Output:""")
            from langchain_core.prompts import ChatPromptTemplate
            promp_template = ChatPromptTemplate.from_messages([
                ('system', s_template), ('human', u_template)
            ])

            from langchain_core.runnables import RunnableLambda

            debug = RunnableLambda(lambda entry:
                                    (LOG.debug(entry), entry)[1])

            parser = CommaSeparatedListOutputParserSet()
            chain = (promp_template
                        | debug
                        | self.model
                        | debug
                        | parser
                        | debug)

            entity_ids = chain.invoke({
                'context': mention.context,
                'sentence': mention.text,
                'term': mention.term,
                'candidates': c_prompt,
            })
            if entity_ids:
                disamb_entities = []
                for entity_id in entity_ids:
                    for c in candidates:
                        c_id = c.id
                        if c_id:
                            if entity_id == c_id:
                                disamb_entities.append(
                                    DisambiguationResult(iri=c.iri))
                                
                return disamb_entities[:limit] if limit else disamb_entities
            raise ValueError(f'Could not disambiguate the term `{mention.term}` among the candidates.')

        except Exception as e:
            logging.warning(f'Exceptions occured while disambiguating the term `{mention.term}`: {e}')
            raise e
