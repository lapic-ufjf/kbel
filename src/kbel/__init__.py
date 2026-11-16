# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

"""KBEL: knowledge base entity linking and knowledge graph construction."""

from kbel.core.candidate import Candidate
from kbel.core.kbel import KBEL
from kbel.core.mention import EntityType, Mention
from kbel.core.results import DisambiguationResult, LinkResult
from kbel.core.triple import Triple
from kbel.coreference import CoreferenceResolver
from kbel.disambiguators import Disambiguator
from kbel.extractors import TripleExtractor

__all__ = [
    'KBEL',
    'Candidate',
    'CoreferenceResolver',
    'Disambiguator',
    'DisambiguationResult',
    'EntityType',
    'LinkResult',
    'Mention',
    'Triple',
    'TripleExtractor',
]
