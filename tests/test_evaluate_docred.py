# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

"""scripts/evaluate_docred.py: Linked-DocRED conversion, run and grid (offline)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from kbel.core.candidate import Candidate
from kbel.core.kbel import KBEL
from kbel.core.results import DisambiguationResult
from kbel.core.triple import Triple
from kbel.disambiguators import Disambiguator
from kbel.extractors import TripleExtractor

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ev = load('evaluate_trex')
dr = load('evaluate_docred')

KB = {'Skai TV': 'Q1', 'Greece': 'Q41', 'country': 'P17', 'Piraeus': 'Q58976'}


class SkaiExtractor(TripleExtractor, strategy_name='test-docred-extract'):
    def __init__(self, strategy_name, **kwargs):
        super().__init__(**kwargs)

    def _extract(self, text, prompt=None):
        return [Triple('Skai TV', 'country', 'Greece')] if 'Skai TV is' in text else []


class First(Disambiguator, strategy_name='test-docred-first'):
    def __init__(self, strategy_name):
        pass

    def _disambiguate(self, mention, candidates, limit, *args, **kwargs):
        return [DisambiguationResult(candidate=candidates[0], score=1.0)]


def entity(i, name, qid=None, type='ORG', sent=0):
    return {'id': i, 'type': type, 'mentions': [{'name': name, 'sent_id': sent, 'pos': [0, 1]},
                                                 {'name': name[:3], 'sent_id': sent, 'pos': [0, 1]}],
            'entity_linking': {'wikidata_resource': qid, 'method': 'manual', 'confidence': 'A'}}


DOC = {
    'title': 'Skai TV',
    'sents': [['Skai', 'TV', 'is', 'a', 'Greek', 'free', '-', 'to', '-', 'air', 'network', '(',
               'founded', '2006', ')', ',', 'based', 'in', 'Piraeus', '.'],
              ['It', 'does', "n't", 'air', 'outside', 'Greece', "'s", 'borders', '.']],
    'entities': [entity(0, 'Skai TV', 'Q1'), entity(1, 'Greece', 'Q41', 'LOC', 1),
                 entity(2, '2006', None, 'TIME'), entity(3, 'Piraeus', 'Q58976', 'LOC'),
                 entity(4, 'Nowhere', None, 'LOC')],
    'relations': [{'r': 'P17', 'h': 0, 't': 1, 'evidence': [0, 1]},
                  {'r': 'P571', 'h': 0, 't': 2, 'evidence': [0]},
                  {'r': 'P159', 'h': 0, 't': 3, 'evidence': []},
                  {'r': 'P131', 'h': 3, 't': 4, 'evidence': [0]},
                  {'r': 'P585', 'h': 2, 't': 0, 'evidence': [0]}],
}


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / 'dev.json'
    path.write_text(json.dumps([DOC, {**DOC, 'title': 'Copy'}]))
    return path


@pytest.fixture(autouse=True)
def fake_search(monkeypatch):
    def lookup(self, mention, limit=None):
        qid = KB.get(mention.term)
        return [Candidate(id=qid, iri=dr.WD + qid, label=mention.term)] if qid else []
    monkeypatch.setattr(KBEL, 'candidates_lookup', lookup)


def test_detokenize():
    assert dr.detokenize(DOC['sents'][0]) == (
        'Skai TV is a Greek free-to-air network (founded 2006), based in Piraeus.')
    assert dr.detokenize(DOC['sents'][1]) == "It doesn't air outside Greece's borders."
    assert dr.detokenize(['He', 'said', '``', 'no', "''", '.']) == 'He said "no".'


def test_convert_text_and_sentence_boundaries(dataset):
    [doc, copy] = dr.load_dataset(dataset)
    assert (doc['docid'], copy['docid']) == ('dev-0000', 'dev-0001')
    (a, b), (c, d) = doc['sentences_boundaries']
    assert doc['text'][a:b].startswith('Skai TV is') and doc['text'][c:d].startswith("It doesn't")


def test_gold_keeps_only_entity_triples_unless_literals_are_included(dataset):
    doc = dr.load_dataset(dataset)[0]
    assert set(ev.gold_triples(doc)) == {('Q1', 'P17', 'Q41'), ('Q1', 'P159', 'Q58976')}
    with_literals = ev.gold_triples(doc, include_literals=True)
    assert ('Q1', 'P571', 'time:2006-01-01') in with_literals and len(with_literals) == 4
    assert ('Q58976', 'P131', 'string:Nowhere') in with_literals
    assert all(g['subject'].startswith('Q') for g in with_literals.values())
    assert ev.gold_triples(doc)[('Q1', 'P17', 'Q41')]['surface'] == ['Skai TV', None, 'Greece']


def test_max_sentences_uses_the_evidence(dataset):
    doc = dr.load_dataset(dataset)[0]
    assert set(ev.gold_triples(doc, max_sentences=1)) == {('Q1', 'P159', 'Q58976')}


def test_distant_supervision_is_refused(tmp_path):
    path = tmp_path / 'train_distant.json'
    path.write_text('[]')
    with pytest.raises(SystemExit, match='distantly supervised'):
        dr.load_dataset(path)


def test_original_docred_format_is_refused(tmp_path):
    path = tmp_path / 'dev.json'
    path.write_text(json.dumps([{'title': 't', 'sents': [], 'vertexSet': [], 'labels': []}]))
    with pytest.raises(SystemExit, match='Linked-DocRED'):
        dr.load_dataset(path)


def test_missing_file_points_to_download(tmp_path):
    with pytest.raises(SystemExit, match='download'):
        dr.load_dataset(tmp_path / 'dev.json')


def run_args(dataset, out):
    return ['run', '--dataset', str(dataset), '-o', str(out), '--kb', 'empty',
            '--extractor', 'test-docred-extract', '--disambiguator', 'test-docred-first',
            '--no-progress', '--no-summary', '--env-file', '/dev/null']


def test_run_on_docred(dataset, tmp_path):
    out = tmp_path / 'run.jsonl'
    assert dr.main(run_args(dataset, out)) == 0
    first, second = [json.loads(line) for line in out.read_text().splitlines()]
    assert first['docid'] == 'dev-0000' and first['status'] == 'ok'
    assert (first['precision'], first['recall']) == (1.0, 0.5)
    assert first['statements'][0]['correct'] is True
    config = json.loads((tmp_path / 'run.jsonl.config.json').read_text())
    assert config['fingerprint']['dataset'] == 'dev.json'


def test_default_output_is_named_after_the_dataset():
    args = ev.build_parser(dataset_name='docred').parse_args(
        ['run', '--extractor', 'relik', '--disambiguator', 'sim', '--output-dir', '/x'])
    assert ev.default_output(args) == Path('/x/docred_wikidata_relik_sim.jsonl')


def test_grid_runs_every_combination(tmp_path, monkeypatch):
    commands = []

    class Done:
        returncode = 0

    def fake_run(command):
        commands.append(command)
        Path(command[command.index('-o') + 1]).write_text('')
        return Done()

    monkeypatch.setattr(ev.subprocess, 'run', fake_run)
    code = dr.main(['grid', '--extractors', 'mrebel,relik', '--disambiguators', 'naive,sim',
                    '--output-dir', str(tmp_path), '--limit', '10', '--max-sentences', '3'])
    assert code == 0 and len(commands) == 4
    first = commands[0]
    assert first[1].endswith('evaluate_docred.py') and first[2] == 'run'
    assert first[-4:] == ['--limit', '10', '--max-sentences', '3']
    pairs = {(c[c.index('--extractor') + 1], c[c.index('--disambiguator') + 1]) for c in commands}
    assert pairs == {('mrebel', 'naive'), ('mrebel', 'sim'), ('relik', 'naive'), ('relik', 'sim')}
    assert (tmp_path / 'docred_wikidata_relik_sim.jsonl').exists()


def test_grid_reports_failures_and_continues(tmp_path, monkeypatch):
    class Result:
        def __init__(self, code):
            self.returncode = code

    monkeypatch.setattr(ev.subprocess, 'run',
                        lambda command: Result(1 if 'mrebel' in command else 0))
    assert dr.main(['grid', '--extractors', 'mrebel,relik', '--disambiguators', 'naive',
                    '--output-dir', str(tmp_path)]) == 1


def test_unknown_arguments_still_fail_outside_grid(dataset, tmp_path):
    with pytest.raises(SystemExit):
        dr.main(run_args(dataset, tmp_path / 'x.jsonl') + ['--nope'])
