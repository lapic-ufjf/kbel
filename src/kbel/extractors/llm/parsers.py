# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

import logging

from langchain_core.output_parsers import BaseOutputParser
from typing_extensions import override

from kbel.core.triple import Triple

LOG = logging.getLogger(__name__)


class TripleOutputParser(BaseOutputParser[list[Triple]]):

    @override
    def parse(self, text: str) -> list[Triple]:
        """Parse the output of an LLM call into a list of Triple objects.

        Args:
            text: The output of an LLM call, one triple per line in the
                format "subject: X | predicate: Y | object: Z".

        Returns:
            A list of Triple objects.
        """
        try:
            triples = []

            for line in text.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue

                fields = {}
                for part in line.split('|'):
                    part = part.strip()
                    if ':' not in part:
                        continue
                    key, value = part.split(':', 1)
                    fields[key.strip().lower()] = value.strip()

                if 'subject' in fields and 'predicate' in fields and 'object' in fields:
                    triples.append(Triple(
                        subject=fields['subject'],
                        predicate=fields['predicate'],
                        object=fields['object'],
                    ))
                else:
                    LOG.debug(f'Skipping malformed triple line: "{line}"')

            return triples
        except Exception as e:
            raise e

    @override
    def get_format_instructions(self) -> str:
        return (
            'Return one triple per line, in the exact format:\n'
            'subject: <subject> | predicate: <predicate> | object: <object>'
        )