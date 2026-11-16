# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import re

from langchain_core.output_parsers import BaseOutputParser
from typing_extensions import override

from kbel.core.triple import Triple

LOG = logging.getLogger(__name__)

_JSON_START = re.compile(r'\s*(?:```(?:json)?\s*)?\[\s*(?:\{|\])')
_BULLET = re.compile(r'^(?:[-*•]|\(?\d+[.)]|\[\d+\])\s+')


class TripleOutputParser(BaseOutputParser[list[Triple]]):
    """Parses triples written one per line or as a JSON list of objects."""

    @staticmethod
    def _from_json(text: str) -> list[Triple] | None:
        start, end = text.find('['), text.rfind(']')
        if start < 0 or end < start:
            return None
        try:
            items = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
        if not isinstance(items, list) or not all(isinstance(i, dict) for i in items):
            return None
        triples = []
        for item in items:
            fields = {str(k).strip().lower(): ' '.join(str(v).split()) for k, v in item.items()}
            if fields.get('subject') and fields.get('predicate') and fields.get('object'):
                triples.append(Triple(fields['subject'], fields['predicate'], fields['object']))
        return triples

    @override
    def parse(self, text: str) -> list[Triple]:
        """Parse the output of an LLM call into a list of Triple objects.

        Args:
            text: The output of an LLM call: one triple per line in the
                format "subject: X | predicate: Y | object: Z", or a JSON
                list of {"subject", "predicate", "object"} objects.

        Returns:
            A list of Triple objects.
        """
        parsed = self._from_json(text) if _JSON_START.match(text) else None
        if parsed is not None:
            return parsed

        triples = []
        for line in text.strip().split('\n'):
            line = _BULLET.sub('', line.strip())
            if not line:
                continue
            fields = {}
            for part in line.split('|'):
                key, sep, value = part.strip().partition(':')
                if sep:
                    fields[key.strip().lower()] = value.strip()
            if fields.get('subject') and fields.get('predicate') and fields.get('object'):
                triples.append(Triple(
                    subject=fields['subject'],
                    predicate=fields['predicate'],
                    object=fields['object'],
                ))
            else:
                LOG.debug(f'Skipping malformed triple line: "{line}"')
        return triples

    @override
    def get_format_instructions(self) -> str:
        return (
            'Return one triple per line, in the exact format:\n'
            'subject: <subject> | predicate: <predicate> | object: <object>'
        )
