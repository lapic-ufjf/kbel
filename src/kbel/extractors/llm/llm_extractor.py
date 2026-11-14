# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

import logging
from textwrap import dedent
from typing import Any, Literal, Optional, TYPE_CHECKING

from kbel.core.triple import Triple

from ..abc import TripleExtractor
from .constants import TE_DEFAULT_EXAMPLES, TE_DEFAULT_PROMPT
from .parsers import TripleOutputParser
from kbel.disambiguators.llm.utils import build_model

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

LOG = logging.getLogger(__name__)


class LLMTripleExtractor(TripleExtractor, strategy_name='llm'):
    """Triple extractor that leverages a large language model (LLM).

    This extractor uses a chat-based LLM to extract subject-predicate-object
    triples from a given text.

    Example:
        >>> extractor = LLMTripleExtractor(
        ...     strategy_name='llm',
        ...     model_name='gpt-4o-mini',
        ...     model_provider='openai',
        ...     model_endpoint='https://api.openai.com/v1',
        ...     model_apikey='LLM_API_KEY'
        ... )
        >>> extractor.extract("Python was created by Guido van Rossum.")
        [Triple(subject='Python', predicate='created by', object='Guido van Rossum')]
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
        """Initializes the LLMTripleExtractor.

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

    def _extract(self, text, prompt = None) -> list[Triple]:
        """Extracts triples from a text using the LLM.

        Args:
            text (str): Input text.

        Returns:
            list[Triple]: List of extracted triples.
        """
        return self.__llm_triple_extraction(text, prompt)

    def __llm_triple_extraction(self, text: str, prompt = None) -> list[Triple]:
        """Internal method that runs the LLM extraction chain.

        Args:
            text (str): Input text.

        Returns:
            list[Triple]: List of extracted triples.

        Raises:
            Exception: Propagates any error raised while invoking the LLM chain.
        """
        try:
            s_template = prompt if prompt is not None else (TE_DEFAULT_PROMPT + '\n\nExamples:\n' + TE_DEFAULT_EXAMPLES)

            u_template = dedent("""Now follow the format strictly.

Input:
    Text: "{text}"

Output:""")

            from langchain_core.prompts import ChatPromptTemplate
            prompt_template = ChatPromptTemplate.from_messages([
                ('system', s_template), ('human', u_template)
            ])

            from langchain_core.runnables import RunnableLambda

            debug = RunnableLambda(lambda entry:
                                    (LOG.debug(entry), entry)[1])

            parser = TripleOutputParser()
            chain = (prompt_template
                        | debug
                        | self.model
                        | debug
                        | parser
                        | debug)

            triples = chain.invoke({'text': text})

            return triples

        except Exception as e:
            LOG.warning(f'Exception occurred while extracting triples: {e}')
            raise e