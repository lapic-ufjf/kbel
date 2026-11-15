# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

"""ReLiK extractors against a fake `relik` package run through the real worker."""

import asyncio
import os
import sys
import textwrap

import pytest

from kbel.core.triple import Triple
from kbel.extractors import TripleExtractor
from kbel.extractors.relik import ReLiKError, ReLiKWorker, worker_client

FAKE_RELIK = '''
import types

class Span:
    def __init__(self, start, end, text, label):
        self.start, self.end, self.text, self.label = start, end, text, label

class Triplets:
    def __init__(self, subject, label, object, confidence):
        self.subject, self.label, self.object, self.confidence = subject, label, object, confidence

class Relik:
    @classmethod
    def from_pretrained(cls, name, **kwargs):
        if name == 'broken':
            raise OSError('no such model')
        print('noise on stdout that must not break the protocol')
        self = cls()
        self.kwargs = kwargs
        return self

    def __call__(self, texts, **options):
        outputs = []
        for i, text in enumerate(texts):
            if 'fail' in text:
                raise ValueError('reader exploded')
            words = text.rstrip('.').split()
            subject = Span(0, len(words[0]), words[0], 'Linked ' + words[0])
            obj = Span(0, len(words[-1]), words[-1], '--NME--')
            confidence = 0.2 if 'weak' in text else 0.9
            outputs.append(types.SimpleNamespace(
                id=i, spans=[subject, obj],
                triplets=[Triplets(subject, options.get('label', 'related to'), obj, confidence),
                          Triplets(subject, options.get('label', 'related to'), obj, confidence)]))
        return outputs[0] if len(outputs) == 1 else list(reversed(outputs))
'''


@pytest.fixture
def fake_relik(tmp_path, monkeypatch):
    package = tmp_path / 'relik'
    package.mkdir()
    (package / '__init__.py').write_text(textwrap.dedent(FAKE_RELIK))
    monkeypatch.setenv('PYTHONPATH', os.pathsep.join(
        [str(tmp_path), os.environ.get('PYTHONPATH', '')]))
    monkeypatch.setenv(worker_client.PYTHON_ENV_VAR, sys.executable)
    return sys.executable


@pytest.fixture
def worker(fake_relik):
    w = ReLiKWorker('fake-model')
    yield w
    w.close()


def test_worker_round_trip_keeps_order(worker):
    outputs = worker.annotate(['Ada met Bob.', 'Carl saw Dan.'])
    assert [o['spans'][0]['text'] for o in outputs] == ['Ada', 'Carl']
    t = outputs[1]['triplets'][0]
    assert t['label'] == 'related to' and t['object']['label'] == '--NME--'
    assert worker.annotate([]) == []


def test_worker_forwards_options_and_errors(worker):
    assert worker.annotate(['Ada met Bob.'], label='knows')[0]['triplets'][0]['label'] == 'knows'
    with pytest.raises(ReLiKError, match='reader exploded'):
        worker.annotate(['This will fail.'])
    assert worker.annotate(['Still alive.'])


def test_worker_reports_load_errors(fake_relik):
    with pytest.raises(ReLiKError, match='no such model'):
        ReLiKWorker('broken')


def test_missing_relik_is_reported(monkeypatch, tmp_path):
    monkeypatch.setenv(worker_client.PYTHON_ENV_VAR, sys.executable)
    monkeypatch.setenv('PYTHONPATH', str(tmp_path))
    with pytest.raises(ReLiKError, match='import relik'):
        worker_client.resolve_python()


def test_no_install_without_permission(monkeypatch, tmp_path):
    monkeypatch.delenv(worker_client.PYTHON_ENV_VAR, raising=False)
    monkeypatch.setattr(worker_client, 'DEFAULT_ENV', tmp_path / 'none')
    with pytest.raises(ReLiKError, match='not installed'):
        worker_client.resolve_python(auto_install=False)


def test_relik_extractor_uses_text_spans(worker):
    e = TripleExtractor('relik', worker=worker)
    assert e.extract('Ada met Bob.') == [Triple('Ada', 'related to', 'Bob')]
    assert e.batch_size == 8


def test_relik_extractor_batch_and_async(worker):
    e = TripleExtractor('relik', worker=worker, batch_size=2)
    texts = ['Ada met Bob.', 'Carl saw Dan.', 'Eve likes Fay.']
    assert asyncio.run(e.aextract_batch(texts)) == [
        [Triple('Ada', 'related to', 'Bob')], [Triple('Carl', 'related to', 'Dan')],
        [Triple('Eve', 'related to', 'Fay')]]


def test_min_confidence(worker):
    e = TripleExtractor('relik', worker=worker, min_confidence=0.5)
    assert e.extract('Ada weak Bob.') == []
    assert e.extract('Ada met Bob.') != []


def test_cie_uses_linked_titles(worker):
    e = TripleExtractor('relik-cie', worker=worker)
    assert e.extract('Ada met Bob.') == [Triple('Linked Ada', 'related to', 'Bob')]


def test_cie_options(worker):
    spans = TripleExtractor('relik-cie', worker=worker, use_entity_titles=False)
    assert spans.extract('Ada met Bob.') == [Triple('Ada', 'related to', 'Bob')]
    linked_only = TripleExtractor('relik-cie', worker=worker, keep_unlinked=False)
    assert linked_only.extract('Ada met Bob.') == []


def test_unknown_option_is_rejected(worker):
    with pytest.raises(TypeError):
        TripleExtractor('relik', worker=worker, use_entity_titles=True)


def test_construct_with_relik(worker, monkeypatch):
    from kbel.core.candidate import Candidate
    from kbel.core.kbel import KBEL
    from kbel.disambiguators import naive  # noqa: F401

    def lookup(self, mention, limit=None):
        return [Candidate(id='Q1', iri='http://www.wikidata.org/entity/Q1', label=mention.term)]

    monkeypatch.setattr(KBEL, 'candidates_lookup', lookup)
    e = TripleExtractor('relik', worker=worker, batch_size=2)
    calls = []
    original = worker.annotate
    monkeypatch.setattr(worker, 'annotate', lambda texts, **kw: calls.append(texts) or original(texts, **kw))
    kbel = KBEL('empty', disambiguator='naive', extractor=e)
    assert len(list(kbel.construct('Ada met Bob. Carl saw Dan. Eve likes Fay.'))) == 1
    assert calls == [['Ada met Bob.', 'Carl saw Dan.'], ['Eve likes Fay.']]


@pytest.mark.integration
def test_real_relik_relation_extraction():
    python = worker_client.env_python(worker_client.DEFAULT_ENV)
    if not (python.exists() and worker_client.has_relik(python)):
        pytest.skip('ReLiK environment not installed')
    e = TripleExtractor('relik', python=python, model_name='relik-ie/relik-relation-extraction-small',
                        auto_install=False)
    try:
        triples = e.extract('Michael Jordan played for the Chicago Bulls in the NBA.')
    finally:
        e.close()
    assert any('Jordan' in t.subject for t in triples)


def test_retriever_runs_without_dataloader_processes(worker, monkeypatch):
    sent = []
    monkeypatch.setattr(worker, 'annotate', lambda texts, **kw: sent.append(kw) or [
        {'spans': [], 'triplets': []} for _ in texts])
    TripleExtractor('relik', worker=worker, top_k=20).extract('Ada met Bob.')
    assert sent == [{'timeout': None, 'top_k': 20, 'num_workers': 0}]
