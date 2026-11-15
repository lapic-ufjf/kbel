# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

"""Light, dependency-free text segmentation (paragraphs and sentences)."""

import re

# One or more blank lines separate paragraphs.
_PARAGRAPH_BOUNDARY = re.compile(r'\n\s*\n')

# [.!?…] (optionally followed by a closing quote/bracket) and whitespace,
# unless the next word starts in lowercase ("e.g. foo", "approx. three").
_SENTENCE_BOUNDARY = re.compile(
    r'(?:(?<=[.!?…])|(?<=[.!?…]["\'”’)\]]))\s+(?![a-z])')


def split_paragraphs(text: str) -> list[str]:
    """Split `text` into paragraphs (blank-line separated), whitespace-normalized."""
    return [' '.join(p.split()) for p in _PARAGRAPH_BOUNDARY.split(text) if p.strip()]


def split_sentences(text: str) -> list[str]:
    """Split `text` into sentences with a regex heuristic.

    Paragraph breaks always end a sentence. Abbreviations followed by a
    capitalized word (e.g. "Dr. Smith") are split; this is a heuristic.
    """
    return [
        s.strip()
        for paragraph in split_paragraphs(text)
        for s in _SENTENCE_BOUNDARY.split(paragraph)
        if s.strip()
    ]
