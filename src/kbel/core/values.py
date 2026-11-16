# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

"""Turns the objects of triples into KIF values that are not entities.

A property of the knowledge base has a datatype (item, time, quantity,
string, ...) that says what its values are. When it is not an item, the
object of a triple is a literal written in the text (a date, a number, a
name), which is parsed here instead of being searched.
"""

import datetime as dt
import re
from decimal import Decimal, InvalidOperation
from typing import Any

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
    Value,
)

SOME_VALUE = 'some'
NO_VALUE = 'none'

SOME_VALUE_MARKERS = frozenset({
    'unknown', 'unknown value', 'some value', 'somevalue', 'an unknown value', 'not known',
    'unknown date', 'unknown number'})
NO_VALUE_MARKERS = frozenset({
    'none', 'no value', 'novalue', 'no one', 'nobody', 'nothing', 'no value exists',
    'does not exist', 'not applicable', 'n/a'})

MONTHS = {name: i for i, names in enumerate((
    ('january', 'jan'), ('february', 'feb'), ('march', 'mar'), ('april', 'apr'), ('may',),
    ('june', 'jun'), ('july', 'jul'), ('august', 'aug'), ('september', 'sep', 'sept'),
    ('october', 'oct'), ('november', 'nov'), ('december', 'dec')), 1) for name in names}

_MONTH = '|'.join(sorted(MONTHS, key=len, reverse=True))
_PREFIX = re.compile(r'^(?:in|on|since|from|until|by|circa|ca|c|about|around|approximately|'
                     r'the year|year)\b\.?\s*', re.IGNORECASE)
_ISO = re.compile(r'^(\d{4})-(\d{1,2})-(\d{1,2})(?:[T ].*)?$')
_SLASH = re.compile(r'^(\d{4})/(\d{1,2})/(\d{1,2})$')
_DMY = re.compile(rf'^(\d{{1,2}})(?:st|nd|rd|th)?(?:\s+of)?\s+({_MONTH})\.?,?\s+(\d{{3,4}})$', re.I)
_MDY = re.compile(rf'^({_MONTH})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{3,4}})$', re.I)
_MY = re.compile(rf'^({_MONTH})\.?,?\s+(\d{{3,4}})$', re.I)
_YEAR = re.compile(r'^(\d{3,4})$')
_DECADE = re.compile(r"^(\d{3})0'?s$")
_ISO_YEAR_MONTH = re.compile(r'^(\d{4})-(\d{1,2})$')

_NUMBER = re.compile(r'^([+-]?)(\d[\d,]*(?:\.\d+)?|\.\d+)\s*(.*)$')
_WORD_NUMBER = re.compile(r'^(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|'
                          r'thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\b\s*(.*)$',
                          re.IGNORECASE)
WORD_NUMBERS = {w: i for i, w in enumerate((
    'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
    'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen',
    'nineteen', 'twenty'))}
MAGNITUDES = {'thousand': 10 ** 3, 'million': 10 ** 6, 'billion': 10 ** 9, 'trillion': 10 ** 12,
              'k': 10 ** 3, 'm': 10 ** 6, 'bn': 10 ** 9}

_QUOTES = ' \t\n"\'“”‘’`'


def clean(text: str) -> str:
    return ' '.join(str(text).strip(_QUOTES).split())


def snak_marker(text: str) -> str | None:
    """`SOME_VALUE` / `NO_VALUE` when the object says the value is unknown / absent."""
    word = clean(text).lower().rstrip('.')
    if word in SOME_VALUE_MARKERS:
        return SOME_VALUE
    if word in NO_VALUE_MARKERS:
        return NO_VALUE
    return None


def _time(year: int, month: int = 1, day: int = 1, precision: Any = None) -> Time | None:
    try:
        moment = dt.datetime(year, month, day, tzinfo=dt.timezone.utc)
    except ValueError:
        return None
    return Time(moment, precision)


def parse_time(text: str) -> Time | None:
    """A KIF `Time` from a date as written in a text, or None.

    Understands ISO dates, ``23 June 1912``, ``June 23, 1912``, ``1st of
    April 2006``, ``June 2010``, ``1903`` and ``1990s``. The precision of
    the result is the one that was written (day, month, year, decade).
    """
    value = _PREFIX.sub('', clean(text)).rstrip('.').strip()
    for pattern, build in (
            (_ISO, lambda m: _time(int(m[1]), int(m[2]), int(m[3]), Time.DAY)),
            (_SLASH, lambda m: _time(int(m[1]), int(m[2]), int(m[3]), Time.DAY)),
            (_DMY, lambda m: _time(int(m[3]), MONTHS[m[2].lower()], int(m[1]), Time.DAY)),
            (_MDY, lambda m: _time(int(m[3]), MONTHS[m[1].lower()], int(m[2]), Time.DAY)),
            (_MY, lambda m: _time(int(m[2]), MONTHS[m[1].lower()], 1, Time.MONTH)),
            (_ISO_YEAR_MONTH, lambda m: _time(int(m[1]), int(m[2]), 1, Time.MONTH)),
            (_YEAR, lambda m: _time(int(m[1]), 1, 1, Time.YEAR)),
            (_DECADE, lambda m: _time(int(m[1]) * 10, 1, 1, Time.DECADE))):
        match = pattern.match(value)
        if match:
            return build(match)
    return None


def parse_quantity(text: str) -> tuple[Decimal, str | None] | None:
    """``(amount, unit text)`` of a number written in a text, or None.

    Understands ``1,234.5``, ``3 million``, ``twelve`` and a trailing unit
    (``78 kilograms``; the unit is returned as text, to be linked).
    """
    value = clean(text)
    rest = ''
    amount: Decimal | None = None
    match = _NUMBER.match(value)
    if match:
        try:
            amount = Decimal((match[1] or '') + match[2].replace(',', ''))
        except InvalidOperation:
            return None
        rest = match[3]
    else:
        word = _WORD_NUMBER.match(value)
        if word:
            amount, rest = Decimal(WORD_NUMBERS[word[1].lower()]), word[2]
    if amount is None:
        return None
    first, _, after = rest.partition(' ')
    if first.lower().rstrip('.') in MAGNITUDES:
        amount *= MAGNITUDES[first.lower().rstrip('.')]
        rest = after
    rest = rest.strip(' .,;')
    if rest == '%':
        rest = 'percent'
    return amount, rest or None


def looks_like_literal(text: str) -> bool:
    """Whether the text reads as a date, a number or a marker of an unknown or absent value.

    Used to start searching an object as an item before its property is
    known, except when it is probably a literal.
    """
    return (snak_marker(text) is not None or parse_time(text) is not None
            or parse_quantity(text) is not None)


def value_class(datatype: Any) -> str:
    """What the objects of a property with this datatype are: ``item``,
    ``property``, ``lexeme`` or ``literal``. An unknown datatype means ``item``."""
    if datatype is None or isinstance(datatype, ItemDatatype):
        return 'item'
    if isinstance(datatype, PropertyDatatype):
        return 'property'
    if isinstance(datatype, LexemeDatatype):
        return 'lexeme'
    return 'literal'


def literal(datatype: Any, text: str, language: str | None = None) -> Value | None:
    """The KIF value of the text `text` for a property of this datatype.

    Returns None when the text is not a valid value of the datatype (e.g. a
    date that cannot be read). Quantities with a unit are not handled here
    (see `parse_quantity`): the unit is an entity that needs linking.
    """
    value = clean(text)
    if not value:
        return None
    if isinstance(datatype, TimeDatatype):
        return parse_time(value)
    if isinstance(datatype, QuantityDatatype):
        parsed = parse_quantity(value)
        return Quantity(parsed[0]) if parsed and parsed[1] is None else None
    if isinstance(datatype, ExternalIdDatatype):
        return ExternalId(value)
    if isinstance(datatype, TextDatatype):
        return Text(value, language or 'en')
    if isinstance(datatype, IRI_Datatype):
        return IRI(value) if re.match(r'^[a-z][a-z0-9+.-]*://\S+$', value, re.IGNORECASE) else None
    if isinstance(datatype, StringDatatype):
        return String(value)
    return None
