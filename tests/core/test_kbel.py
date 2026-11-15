# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

import pytest

from kbel.core.candidate import Candidate
from kbel.core.kbel import KBEL
from kbel.core.mention import EntityType, Mention
from kbel.core.triple import Triple
from kbel.disambiguators import naive  # noqa: F401  (registers "naive")
from kbel.extractors import TripleExtractor


@pytest.fixture
def linker(monkeypatch):
    """KBEL on the offline `empty` search, with a scripted candidate list."""
    kb = KBEL('empty', disambiguator='naive')
    found: dict[str, list[Candidate]] = {}
    monkeypatch.setattr(kb, 'candidates_lookup',
                        lambda mention, limit=None: found.get(mention.term, []))
    kb.found = found
    return kb


def test_link_without_candidates_returns_empty_list(linker):
    assert linker.link(Mention(term='nothing', text='')) == []


def test_link_returns_disambiguated_entity(linker):
    linker.found['Python'] = [
        Candidate(iri='http://www.wikidata.org/entity/Q271218', label='Python genus'),
        Candidate(iri='http://www.wikidata.org/entity/Q28865', label='Python'),
    ]
    [result] = linker.link(Mention(term='Python', text=''))
    assert result.candidate.iri.endswith('Q28865')


class FixedExtractor(TripleExtractor, strategy_name='test-fixed'):
    def __init__(self, strategy_name, **kwargs):
        super().__init__(**kwargs)

    def _extract(self, text, prompt=None):
        return [Triple('She', 'was born in', 'Warsaw')]


def test_extract_and_link_tolerates_unlinkable_parts(linker):
    linker.extractor = 'test-fixed'
    linker.found['Warsaw'] = [Candidate(iri='http://www.wikidata.org/entity/Q270', label='Warsaw')]

    [(triple, (subject, predicate, obj))] = linker.extract_and_link('She was born in Warsaw.')
    assert triple == Triple('She', 'was born in', 'Warsaw')
    assert subject == [] and predicate == []
    assert obj[0].candidate.label == 'Warsaw'


def test_link_requires_disambiguator():
    with pytest.raises(RuntimeError, match='No disambiguator'):
        KBEL('empty').link(Mention(term='x', text='', entity_type=EntityType.ITEM))


def test_extract_requires_extractor():
    with pytest.raises(RuntimeError, match='No extractor'):
        KBEL('empty').extract('Some text.')
