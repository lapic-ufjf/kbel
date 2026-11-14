# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from kif_lib import IRI, Text
from typing import (
    Any,
    cast,
    ClassVar,
    Iterator,
    Literal,
    Mapping,
    override,
    TypeVar,
)
from kif_lib.search.wikidata_wapi import WikidataWAPI_Search, WikidataWAPI_SearchOptions


TOptions = TypeVar(
    'TOptions',
    bound=WikidataWAPI_SearchOptions,
    default=WikidataWAPI_SearchOptions,
)


class WikidataVDB_Search(
        WikidataWAPI_Search,
        search_name='wikidata-vdb',
        search_description='Wikidata Vector Database search'
):
    """Wikidata search using the Wikidata Vector Database."""

    PROPERTY_ENDPOINT: ClassVar[str] = (
        'https://wd-vectordb.wmcloud.org/property/query/'
    )

    ITEM_ENDPOINT: ClassVar[str] = (
        'https://wd-vectordb.wmcloud.org/item/query/'
    )

    WIKIDATA_ENDPOINT: ClassVar[str] = (
        'https://www.wikidata.org/w/api.php'
    )

    METADATA_BATCH_SIZE: ClassVar[int] = 50

    def __init__(
            self,
            search_name: str,
            *args: Any,
            rerank: bool = False,
            **kwargs: Any
    ) -> None:
        self._rerank = rerank
        super().__init__(search_name, *args, **kwargs)

    @override
    def _item_data(
            self,
            search: str,
            options: TOptions
    ) -> Iterator[WikidataVDB_Search.TData]:
        return self._x_data('item', search, options)

    @override
    def _property_data(
            self,
            search: str,
            options: TOptions
    ) -> Iterator[WikidataVDB_Search.TData]:
        return self._x_data('property', search, options)

    @override
    def _check_options(
        self,
        type: Literal['item', 'lexeme', 'property'],
        options: TOptions
    ) -> tuple[IRI, str, int, int, float | None]:

        _, language, limit, page_size, timeout = super()._check_options(
            type,
            options
        )

        iri = IRI(
            self.ITEM_ENDPOINT
            if type == 'item'
            else self.PROPERTY_ENDPOINT
        )

        return iri, language, limit, page_size, timeout

    @override
    def _x_data(
        self,
        type: Literal['item', 'lexeme', 'property'],
        search: str,
        options: TOptions
    ) -> Iterator[WikidataVDB_Search.TData]:

        iri, language, limit, _, timeout = self._check_options(
            type,
            options
        )

        params = {
            'query': search,
            'lang': language,
            'rerank': str(self._rerank).lower(),
        }

        response = self._http_get_json(
            iri.content,
            params=params,
            timeout=timeout
        )

        results = cast(
            list[Mapping[str, Any]],
            response
        )

        if limit is None:
            limit = options.max_limit

        results = results[:limit]

        ids: list[str] = []

        for result in results:
            entity_id = self._get_entity_id(result, type)

            if entity_id is not None:
                ids.append(entity_id)

        if not ids:
            return

        metadata = self._get_wikidata_entities(
            ids,
            language,
            timeout
        )

        for result in results:
            entity_id = self._get_entity_id(result, type)

            if entity_id is None:
                continue

            entity = metadata.get(entity_id)

            if entity is None:
                continue

            yield self._normalize_entity(
                entity,
                result,
                type,
                language
            )

    def _get_entity_id(
            self,
            result: Mapping[str, Any],
            type: Literal['item', 'lexeme', 'property']
    ) -> str | None:

        if type == 'property':
            return result.get('PID')

        return result.get('QID')

    def _get_wikidata_entities(
        self,
        ids: list[str],
        language: str,
        timeout: float | None
    ) -> dict[str, dict[str, Any]]:

        entities: dict[str, dict[str, Any]] = {}

        for i in range(0, len(ids), self.METADATA_BATCH_SIZE):
            batch = ids[i:i + self.METADATA_BATCH_SIZE]

            params = {
                'action': 'wbgetentities',
                'format': 'json',
                'ids': '|'.join(batch),
                'props': 'labels|descriptions|aliases|claims',
                'languages': language,
                'languagefallback': '1',
                'origin': '*',
            }

            response = self._http_get_json(
                self.WIKIDATA_ENDPOINT,
                params=params,
                timeout=timeout
            )

            for entity_id, entity in response.get(
                'entities', {}
            ).items():

                if 'missing' not in entity:
                    entities[entity_id] = entity

        return entities

    def _normalize_entity(
        self,
        entity: Mapping[str, Any],
        result: Mapping[str, Any],
        type: Literal['item', 'lexeme', 'property'],
        language: str
    ) -> WikidataVDB_Search.TData:

        entity_id = entity['id']

        data: WikidataVDB_Search.TData = {
            'id': entity_id,
            'concepturi': (
                f'https://www.wikidata.org/entity/{entity_id}'
            ),
            'language': language,
        }

        if type == 'property':
            data['datatype'] = entity.get(
                'datatype',
                'http://wikiba.se/ontology#String'
            )

        labels = entity.get('labels', {})

        if language in labels:
            data['label'] = labels[language]['value']

        descriptions = entity.get('descriptions', {})

        if language in descriptions:
            data['description'] = descriptions[language]['value']

        aliases = entity.get('aliases', {})
        if language in aliases:
            data['aliases'] = [
                alias['value']
                for alias in aliases[language]
            ]

        match = result.get('similarity_score')

        if match is not None:
            data['similarity_score'] = match

        rrf_score = result.get('rrf_score')

        if rrf_score is not None:
            data['rrf_score'] = rrf_score

        source = result.get('source')

        if source is not None:
            data['source'] = source

        return data

    @override
    def _to_x_descriptor(
        self,
        data: WikidataVDB_Search.TData,
        empty: dict[str, Any] | None = None
    ) -> dict[str, Any]:

        if empty is None:
            empty = {}

        try:
            language = data.get('language', 'en')

            descriptor: dict[str, Any] = {}

            if 'label' in data:
                descriptor['labels'] = {
                    language: Text(
                        data['label'],
                        language
                    )
                }

            if 'description' in data:
                descriptor['descriptions'] = {
                    language: Text(
                        data['description'],
                        language
                    )
                }

            if 'aliases' in data:
                descriptor['aliases'] = {
                    language: {
                        Text(alias, language)
                        for alias in data['aliases']
                    }
                }

            return descriptor

        except KeyError:
            return empty
