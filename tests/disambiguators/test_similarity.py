# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

"""The `sim` disambiguator with a deterministic fake embedding model."""

import numpy as np
import pytest

pytest.importorskip('sentence_transformers')

from kbel.core.candidate import Candidate  # noqa: E402
from kbel.core.mention import Mention  # noqa: E402
from kbel.disambiguators import Disambiguator  # noqa: E402
from kbel.disambiguators import similarity  # noqa: E402,F401

AXES = ['snake', 'code', 'film', 'city']


class FakeModel:
    def __init__(self):
        self.queries, self.documents = [], []

    @staticmethod
    def _vec(text):
        v = np.array([text.lower().count(a) for a in AXES], dtype=np.float32) + 1e-3
        return v / np.linalg.norm(v)

    def encode_query(self, texts, normalize_embeddings=True):
        self.queries.extend(texts)
        return np.stack([self._vec(t) for t in texts])

    def encode_document(self, texts, normalize_embeddings=True):
        self.documents.extend(texts)
        return np.stack([self._vec(t) for t in texts])

    encode = encode_query


SNAKE = Candidate(id='Q2', iri='wd:Q2', label='Python', description='snake genus')
CODE = Candidate(id='Q1', iri='wd:Q1', label='Python', description='code language')
FILM = Candidate(id='Q3', iri='wd:Q3', label='Python (film)', description=None)


def sim(**kw):
    return Disambiguator('sim', model=FakeModel(), **kw)


def test_sentence_decides():
    d = sim()
    [r] = d.disambiguate(Mention(term='Python', text='Python code is clean.'), [SNAKE, CODE], 1)
    assert r.candidate is CODE


def test_context_breaks_ties_the_sentence_cannot():
    d = sim(rank_weight=0)
    mention = Mention(term='Python', text='Python is popular.',
                      context='Python is popular. Its code runs everywhere, code code.')
    assert d.disambiguate(mention, [SNAKE, CODE], 1)[0].candidate is CODE
    no_context = sim(rank_weight=0, context_weight=0)
    assert no_context.disambiguate(mention, [SNAKE, CODE], 1)[0].candidate is SNAKE


def test_context_equal_to_the_sentence_is_not_encoded_twice():
    d = sim()
    d.disambiguate(Mention(term='Python', text='Python code.', context='Python code.'), [CODE], 1)
    assert d.model.queries == ['Python code.']


def test_search_rank_is_a_prior():
    d = sim(sentence_weight=0, context_weight=0, exact_match_weight=0)
    results = d.disambiguate(Mention(term='Python', text='x'), [SNAKE, CODE, FILM], 3)
    assert [r.candidate for r in results] == [SNAKE, CODE, FILM]
    assert results[0].score > results[1].score > results[2].score


def test_exact_label_match_bonus_and_missing_description():
    d = sim(sentence_weight=0, context_weight=0, rank_weight=0)
    [r] = d.disambiguate(Mention(term='python (FILM)', text='x'), [SNAKE, FILM], 1)
    assert r.candidate is FILM
    assert 'Python (film)' in d.model.documents


def test_term_weight_compares_term_and_label():
    d = sim(sentence_weight=0, context_weight=0, rank_weight=0, exact_match_weight=0, term_weight=1)
    city = Candidate(id='Q4', iri='wd:Q4', label='city', description='')
    [r] = d.disambiguate(Mention(term='big city', text='x'), [SNAKE, city], 1)
    assert r.candidate is city


def test_embeddings_are_cached():
    d = sim()
    mention = Mention(term='Python', text='Python code.', context='A paragraph about code. Python code.')
    for _ in range(3):
        d.disambiguate(mention, [SNAKE, CODE], 1)
    assert len(d.model.queries) == 2 and len(d.model.documents) == 2


def test_cache_is_bounded():
    d = Disambiguator('sim', model=FakeModel(), cache_size=2)
    for i in range(5):
        d.disambiguate(Mention(term='t', text=f'sentence {i}'), [CODE], 1)
    assert len(d._queries._items) <= 2


def test_limit_and_scores_sorted():
    results = sim().disambiguate(Mention(term='Python', text='snake'), [SNAKE, CODE, FILM], 2)
    assert len(results) == 2 and results[0].score >= results[1].score
