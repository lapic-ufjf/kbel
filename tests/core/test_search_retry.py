# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

"""Transient knowledge-base failures are retried, never turned into "no candidates"."""

import httpx
import pytest
from kif_lib import Item, Text

from kbel.core import kbel as kbel_module
from kbel.core.kbel import KBEL
from kbel.core.mention import EntityType, Mention

MENTION = Mention(term='Python', text='', entity_type=EntityType.ITEM, language='en')


def status_error(code, headers=None):
    request = httpx.Request('GET', 'https://www.wikidata.org/w/api.php')
    return httpx.HTTPStatusError(
        f'{code}', request=request, response=httpx.Response(code, headers=headers, request=request))


class FlakyKB:
    def __init__(self, errors):
        self.errors = list(errors)
        self.calls = 0

    def item_descriptor(self, search, limit):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        yield Item('http://www.wikidata.org/entity/Q28865'), {
            'labels': {'en': Text('Python')}, 'descriptions': {'en': Text('language')}}


@pytest.fixture
def no_sleep(monkeypatch):
    waits = []
    monkeypatch.setattr(kbel_module.time, 'sleep', waits.append)
    return waits


def test_rate_limit_is_retried(no_sleep):
    kb = FlakyKB([status_error(429), status_error(503)])
    [candidate] = KBEL(kb).candidates_lookup(MENTION)
    assert candidate.label == 'Python' and kb.calls == 3
    assert len(no_sleep) == 2 and no_sleep[1] > no_sleep[0]


def test_retry_after_header_is_honoured(no_sleep):
    KBEL(FlakyKB([status_error(429, {'Retry-After': '7'})])).candidates_lookup(MENTION)
    assert 7 <= no_sleep[0] <= 7 * 1.5


def test_network_errors_are_retried(no_sleep):
    kb = FlakyKB([httpx.ConnectError('down')])
    assert KBEL(kb).candidates_lookup(MENTION)


def test_persistent_rate_limit_raises_instead_of_returning_nothing(no_sleep):
    linker = KBEL(FlakyKB([status_error(429)] * 10))
    linker.search_retries = 2
    with pytest.raises(httpx.HTTPStatusError):
        linker.candidates_lookup(MENTION)
    assert len(no_sleep) == 2


def test_other_errors_are_skipped_without_retrying(no_sleep):
    kb = FlakyKB([status_error(404)])
    assert KBEL(kb).candidates_lookup(MENTION) == []
    assert kb.calls == 1 and no_sleep == []
