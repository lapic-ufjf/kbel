# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

from decimal import Decimal

import pytest
from kif_lib import (
    IRI,
    ExternalId,
    ExternalIdDatatype,
    IRI_Datatype,
    ItemDatatype,
    LexemeDatatype,
    PropertyDatatype,
    Quantity,
    QuantityDatatype,
    String,
    StringDatatype,
    Text,
    TextDatatype,
    Time,
    TimeDatatype,
)

from kbel.core.values import (
    NO_VALUE,
    SOME_VALUE,
    literal,
    looks_like_literal,
    parse_quantity,
    parse_time,
    snak_marker,
    value_class,
)


@pytest.mark.parametrize('text,day,precision', [
    ('23 June 1912', '1912-06-23', Time.DAY),
    ('June 23, 1912', '1912-06-23', Time.DAY),
    ('23rd June 1912', '1912-06-23', Time.DAY),
    ('1st of April 2006', '2006-04-01', Time.DAY),
    ('April 1, 2006', '2006-04-01', Time.DAY),
    ('Sept. 3, 1999', '1999-09-03', Time.DAY),
    ('June 2010', '2010-06-01', Time.MONTH),
    ('2010-06', '2010-06-01', Time.MONTH),
    ('1903', '1903-01-01', Time.YEAR),
    ('in 1815', '1815-01-01', Time.YEAR),
    ('circa 1815', '1815-01-01', Time.YEAR),
    ('1990s', '1990-01-01', Time.DECADE),
    ('1815-12-10', '1815-12-10', Time.DAY),
    ('1815-12-10T00:00:00Z', '1815-12-10', Time.DAY),
    ('2014/03/11', '2014-03-11', Time.DAY),
    ('"23 June 1912"', '1912-06-23', Time.DAY),
])
def test_parse_time(text, day, precision):
    value = parse_time(text)
    assert value.time.date().isoformat() == day
    assert value.precision == precision


@pytest.mark.parametrize('text', ['yesterday', 'June', '31 February 2000', 'Alan', '', '12'])
def test_parse_time_rejects_what_is_not_a_date(text):
    assert parse_time(text) is None


@pytest.mark.parametrize('text,amount,unit', [
    ('78', '78', None), ('1,234.5', '1234.5', None), ('-3', '-3', None),
    ('3 million', '3000000', None), ('2.5 billion', '2500000000', None),
    ('twelve', '12', None), ('78 kilograms', '78', 'kilograms'),
    ('50%', '50', 'percent'), ('twelve research laboratories', '12', 'research laboratories'),
])
def test_parse_quantity(text, amount, unit):
    assert parse_quantity(text) == (Decimal(amount), unit)


def test_parse_quantity_rejects_text():
    assert parse_quantity('abc') is None
    assert parse_quantity('') is None


@pytest.mark.parametrize('text,expected', [
    ('unknown', SOME_VALUE), ('Unknown value.', SOME_VALUE), ('"some value"', SOME_VALUE),
    ('none', NO_VALUE), ('No value', NO_VALUE), ('nobody', NO_VALUE), ('n/a', NO_VALUE),
    ('Paris', None), ('unknown soldier', None), ('', None),
])
def test_snak_marker(text, expected):
    assert snak_marker(text) == expected


@pytest.mark.parametrize('datatype,kind', [
    (None, 'item'), (ItemDatatype(), 'item'), (PropertyDatatype(), 'property'),
    (LexemeDatatype(), 'lexeme'), (TimeDatatype(), 'literal'), (QuantityDatatype(), 'literal'),
    (StringDatatype(), 'literal'), (TextDatatype(), 'literal'), (IRI_Datatype(), 'literal'),
    (ExternalIdDatatype(), 'literal'),
])
def test_value_class(datatype, kind):
    assert value_class(datatype) == kind


def test_literal_builds_the_value_of_each_datatype():
    assert literal(StringDatatype(), ' "Ada" ') == String('Ada')
    assert literal(ExternalIdDatatype(), 'tt0111161') == ExternalId('tt0111161')
    assert literal(TextDatatype(), 'Ada', 'pt') == Text('Ada', 'pt')
    assert literal(TextDatatype(), 'Ada') == Text('Ada', 'en')
    assert literal(IRI_Datatype(), 'https://example.org/a') == IRI('https://example.org/a')
    assert literal(IRI_Datatype(), 'not a url') is None
    assert literal(QuantityDatatype(), '1,000') == Quantity(1000)
    assert literal(QuantityDatatype(), '5 km') is None
    assert literal(TimeDatatype(), '1903').time.year == 1903
    assert literal(TimeDatatype(), 'soon') is None
    assert literal(StringDatatype(), '  ') is None
    assert literal(ItemDatatype(), 'Paris') is None


def test_looks_like_literal():
    assert looks_like_literal('1903') and looks_like_literal('12 books')
    assert looks_like_literal('unknown') and looks_like_literal('June 2010')
    assert not looks_like_literal('Warsaw') and not looks_like_literal('Marie Curie')
