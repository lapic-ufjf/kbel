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


def test_extract_and_link_survives_a_failing_link(linker, monkeypatch):
    linker.extractor = 'test-fixed'
    linker.found['Warsaw'] = [Candidate(iri='http://www.wikidata.org/entity/Q270', label='Warsaw')]
    linker.found['She'] = [Candidate(iri='http://www.wikidata.org/entity/Q1', label='She')]
    disambiguate = linker.disambiguator._disambiguate

    def flaky(mention, candidates, limit, *args, **kwargs):
        if mention.term == 'She':
            raise ValueError('model down')
        return disambiguate(mention, candidates, limit)

    monkeypatch.setattr(linker.disambiguator, '_disambiguate', flaky)
    [(_, (subject, _, obj))] = linker.extract_and_link('She was born in Warsaw.')
    assert subject == []
    assert obj[0].candidate.label == 'Warsaw'


def test_public_api_is_importable_from_the_package():
    import kbel

    for name in ('KBEL', 'Mention', 'EntityType', 'Candidate', 'Triple', 'LinkResult',
                 'DisambiguationResult', 'Disambiguator', 'TripleExtractor'):
        assert getattr(kbel, name) is not None and name in kbel.__all__
    assert kbel.KBEL is KBEL


def test_builtin_plugins_are_imported_on_first_use():
    import subprocess
    import sys

    code = ("from kbel import KBEL, Disambiguator\n"
            "assert 'naive' not in Disambiguator.registry\n"
            "kb = KBEL('empty', disambiguator='naive')\n"
            "assert type(kb.disambiguator).__name__ == 'NaiveDisambiguator'\n"
            "try:\n    Disambiguator('no-such-plugin')\n"
            "except ValueError as e:\n    assert 'no such disambiguator' in str(e)\n"
            "else:\n    raise SystemExit('unknown plugin accepted')\n")
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_every_builtin_plugin_name_points_to_a_module():
    import importlib.util

    from kbel.disambiguators import Disambiguator

    for module in {*Disambiguator.builtin.values(), *TripleExtractor.builtin.values()}:
        assert importlib.util.find_spec(module.rsplit('.', 1)[0]) is not None
        package, _, name = module.rpartition('.')
        root = importlib.util.find_spec(package).submodule_search_locations[0]
        from pathlib import Path
        assert (Path(root) / f'{name}.py').exists() or (Path(root) / name / '__init__.py').exists()
