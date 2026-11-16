# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

"""`construct` builds every kind of KIF statement: items, literals and snaks."""

import pytest
from kif_lib import (
    ExternalId,
    Item,
    ItemDatatype,
    NoValueSnak,
    Property,
    PropertyDatatype,
    Quantity,
    QuantityDatatype,
    SomeValueSnak,
    String,
    StringDatatype,
    Text,
    TextDatatype,
    Time,
    TimeDatatype,
    ValueSnak,
)

from kbel.core.candidate import Candidate
from kbel.core.kbel import KBEL
from kbel.core.mention import EntityType, Mention
from kbel.core.results import DisambiguationResult
from kbel.core.triple import Triple
from kbel.disambiguators import Disambiguator
from kbel.extractors import TripleExtractor

WD = 'http://www.wikidata.org/entity/'
ITEMS = {'Ada': 'Q7259', 'London': 'Q84', 'Babbage': 'Q46633', 'kilogram': 'Q11570',
         'Birth': 'Q1', 'Analytical Engine': 'Q2'}
PROPERTIES = {
    'place of birth': ('P19', ItemDatatype()),
    'date of birth': ('P569', TimeDatatype()),
    'population': ('P1082', QuantityDatatype()),
    'mass': ('P2067', QuantityDatatype()),
    'also known as name': ('P742', StringDatatype()),
    'native label': ('P1705', TextDatatype()),
    'imdb id': ('P345', None),
    'subclass of': ('P279', PropertyDatatype()),
    'child': ('P40', ItemDatatype()),
    'date of death': ('P570', TimeDatatype()),
    'mystery': ('P9999', None),
}
PROPERTIES['imdb id'] = ('P345', None)

TRIPLES = {
    'a': [Triple('Ada', 'place of birth', 'London')],
    'b': [Triple('Ada', 'date of birth', '10 December 1815')],
    'c': [Triple('Ada', 'population', '3 million')],
    'd': [Triple('Ada', 'mass', '60 kilogram')],
    'e': [Triple('Ada', 'also known as name', 'Lady Lovelace')],
    'f': [Triple('Ada', 'native label', 'Ada Lovelace')],
    'g': [Triple('Ada', 'date of death', 'unknown')],
    'h': [Triple('Ada', 'child', 'none')],
    'i': [Triple('Ada', 'date of birth', 'soon')],
    'j': [Triple('Ada', 'mystery', 'London')],
    'k': [Triple('Ada', 'subclass of', 'place of birth')],
    'l': [Triple('Ada', 'population', '60 unobtainium')],
}


class Script(TripleExtractor, strategy_name='test-statements-script'):
    def __init__(self, strategy_name):
        super().__init__()

    def _extract(self, text, prompt=None):
        return list(TRIPLES.get(text.rstrip('.'), []))


class First(Disambiguator, strategy_name='test-statements-first'):
    def __init__(self, strategy_name):
        self.mentions = []

    def _disambiguate(self, mention, candidates, limit, *args, **kwargs):
        self.mentions.append(mention)
        return [DisambiguationResult(candidate=candidates[0], score=1.0)]


@pytest.fixture
def kbel(monkeypatch):
    searches = []

    def lookup(self, mention, limit=None):
        searches.append((mention.term, mention.entity_type))
        if mention.entity_type is EntityType.PROPERTY:
            found = PROPERTIES.get(mention.term)
            if found is None:
                return []
            pid, datatype = found
            return [Candidate(iri=WD + pid, id=pid, label=mention.term, datatype=datatype)]
        qid = ITEMS.get(mention.term)
        return [Candidate(iri=WD + qid, id=qid, label=mention.term)] if qid else []

    monkeypatch.setattr(KBEL, 'candidates_lookup', lookup)
    k = KBEL('empty', disambiguator=Disambiguator('test-statements-first'),
             extractor=TripleExtractor('test-statements-script'))
    k.searches = searches
    return k


def construct(kbel, key):
    return list(kbel.construct(f'{key}.'))


def snak_of(kbel, key):
    [statement] = construct(kbel, key)
    assert statement.subject == Item(WD + 'Q7259')
    return statement.snak


def test_item_values_are_value_snaks_with_items(kbel):
    snak = snak_of(kbel, 'a')
    assert snak == ValueSnak(Property(WD + 'P19', ItemDatatype()), Item(WD + 'Q84'))


def test_dates_become_times_without_searching_them(kbel):
    snak = snak_of(kbel, 'b')
    assert isinstance(snak, ValueSnak) and isinstance(snak.value, Time)
    assert snak.value.time.date().isoformat() == '1815-12-10'
    assert snak.value.precision == Time.DAY
    assert snak.property == Property(WD + 'P569', TimeDatatype())
    assert ('10 December 1815', EntityType.ITEM) not in kbel.searches


def test_numbers_become_quantities(kbel):
    assert snak_of(kbel, 'c').value == Quantity(3000000)
    assert ('3 million', EntityType.ITEM) not in kbel.searches


def test_a_quantity_with_a_unit_links_the_unit(kbel):
    snak = snak_of(kbel, 'd')
    assert snak.value == Quantity(60, Item(WD + 'Q11570'))
    assert ('kilogram', EntityType.ITEM) in kbel.searches


def test_a_quantity_with_an_unknown_unit_is_skipped(kbel):
    assert construct(kbel, 'l') == []


@pytest.mark.parametrize('key,expected', [
    ('e', String('Lady Lovelace')), ('f', Text('Ada Lovelace', 'en'))])
def test_strings_and_texts(kbel, key, expected):
    assert snak_of(kbel, key).value == expected


def test_an_unknown_value_is_a_some_value_snak(kbel):
    snak = snak_of(kbel, 'g')
    assert snak == SomeValueSnak(Property(WD + 'P570', TimeDatatype()))
    assert ('unknown', EntityType.ITEM) not in kbel.searches


def test_no_value_is_a_no_value_snak(kbel):
    assert snak_of(kbel, 'h') == NoValueSnak(Property(WD + 'P40', ItemDatatype()))


def test_unreadable_literals_are_skipped(kbel):
    assert construct(kbel, 'i') == []


def test_an_unknown_datatype_means_an_item(kbel):
    snak = snak_of(kbel, 'j')
    assert snak.value == Item(WD + 'Q84')


def test_a_property_valued_property_links_the_object_as_a_property(kbel):
    snak = snak_of(kbel, 'k')
    assert snak.value == Property(WD + 'P19', ItemDatatype())
    assert ('place of birth', EntityType.PROPERTY) in kbel.searches


def test_extract_and_link_exposes_literals_and_markers(kbel):
    [(_, (s, p, o))] = list(kbel.extract_and_link('b.', limit=3))
    assert [r.entity for r in s] == [Item(WD + 'Q7259')]
    assert p[0].entity == Property(WD + 'P569', TimeDatatype())
    [literal] = o
    assert literal.is_literal and literal.candidate is None
    assert isinstance(literal.entity, Time)
    [(_, (_, _, o))] = list(kbel.extract_and_link('g.'))
    assert o == []
    [(_, (_, _, o))] = list(kbel.extract_and_link('a.'))
    assert not o[0].is_literal and o[0].candidate.label == 'London'


def test_statements_are_deduplicated_and_ordered(kbel):
    TRIPLES['m'] = [Triple('Ada', 'date of birth', '1815-12-10'),
                    Triple('Ada', 'date of birth', '10 December 1815')]
    try:
        assert len(construct(kbel, 'm')) == 1
    finally:
        del TRIPLES['m']


def test_statements_survive_a_failing_unit_link(kbel):
    kbel.disambiguator = Disambiguator('test-statements-first')
    original = kbel.disambiguator._disambiguate

    def fail_for_units(mention, candidates, limit, *a, **k):
        if mention.term == 'kilogram':
            raise ValueError('boom')
        return original(mention, candidates, limit)

    kbel.disambiguator._disambiguate = fail_for_units
    assert construct(kbel, 'd') == []
    assert len(construct(kbel, 'a')) == 1
