# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal, TYPE_CHECKING


if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


def build_model(
    model_name: str,
    provider: Literal['ibm', 'openai', 'ollama'],
    endpoint: str,
    apikey: str | None = None,
    **kwargs) -> "BaseChatModel":
    """Builds a LangChain chat model; `kwargs` are the model's parameters.

    `apikey` is required by every provider except ``ollama``.
    """
    if not model_name or not endpoint:
        raise ValueError('`model_name` and `model_endpoint` are required to build the model.')
    if provider not in ('ibm', 'openai', 'ollama'):
        raise ValueError(f'Unknown model provider `{provider}` (ibm, openai or ollama).')
    if provider != 'ollama' and not apikey:
        raise ValueError(f'`model_apikey` is required by the `{provider}` provider.')

    if provider == 'ibm':
        kwargs = dict(kwargs)
        project_id = kwargs.pop('project_id', None)
        if not project_id:
            raise ValueError('IBM models need `project_id` in `model_params`.')
        try:
            from langchain_ibm import ChatWatsonx
        except ImportError as err:
            raise ImportError(f'{__name__} requires langchain_ibm') from err
        return ChatWatsonx(
            model_id=model_name,
            apikey=apikey,  # type: ignore
            url=endpoint,  # type: ignore
            project_id=project_id,
            params=kwargs)
    if provider == 'openai':
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as err:
            raise ImportError(f'{__name__} requires langchain_openai') from err
        return ChatOpenAI(
            model=model_name,
            api_key=apikey,  # type: ignore
            base_url=endpoint,  # type: ignore
            **kwargs)
    try:
        from langchain_ollama import ChatOllama
    except ImportError as err:
        raise ImportError(f'{__name__} requires langchain_ollama') from err
    return ChatOllama(model=model_name, base_url=endpoint, **kwargs)
