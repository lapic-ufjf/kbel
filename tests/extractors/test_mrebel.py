# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

"""mREBEL batching and parsing, without loading the model."""

import threading

from kbel.core.triple import Triple
from kbel.extractors.mrebel import MREBELTripleExtractor


def fake_extractor(batch_size=2, beams=2):
    e = object.__new__(MREBELTripleExtractor)
    e.batch_size = batch_size
    e._num_return_sequences = beams
    e._split_sentences = True
    e._graph = False
    e._lock = threading.Lock()
    e.calls = []

    def generate(texts):
        e.calls.append(list(texts))
        return [[f'<s><triplet> {t.rstrip(".")} <per> x <loc> rel{b}</s>' for b in range(beams)]
                for t in texts]

    e._generate = generate
    return e


def test_batches_group_texts_by_length_and_keep_order():
    e = fake_extractor()
    texts = ['Bbbbbbbbbb.', 'A.', 'Cccccc.']
    result = e.extract_batch(texts)
    assert e.calls == [['A.', 'Cccccc.'], ['Bbbbbbbbbb.']]
    assert [r[0].subject for r in result] == ['Bbbbbbbbbb', 'A', 'Cccccc']
    assert result[1] == [Triple('A', 'rel0', 'x'), Triple('A', 'rel1', 'x')]


def test_extract_splits_sentences_and_dedupes():
    e = fake_extractor(beams=1)
    triples = e.extract('Ada met Bob. Ada met Bob. Carl ran.')
    assert e.calls == [['Carl ran.', 'Ada met Bob.'], ['Ada met Bob.']]
    assert triples == [Triple('Ada met Bob', 'rel0', 'x'), Triple('Carl ran', 'rel0', 'x')]


def test_parse_multiple_objects_of_one_subject():
    seq = '<s><triplet> Python <concept> Guido <per> designed by <concept> 1991 <date> inception</s>'
    assert MREBELTripleExtractor._parse_triplets(seq) == [
        {'subject': 'Python', 'predicate': 'designed by', 'object': 'Guido',
         'subject_type': 'concept', 'object_type': 'per'},
        {'subject': 'Python', 'predicate': 'inception', 'object': '1991',
         'subject_type': 'concept', 'object_type': 'date'}]
