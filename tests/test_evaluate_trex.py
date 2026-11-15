# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

"""scripts/evaluate_trex.py, offline: fake plugins and a tiny T-REx file."""

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
_spec = importlib.util.spec_from_file_location('evaluate_trex', ROOT / 'scripts' / 'evaluate_trex.py')
ev = importlib.util.module_from_spec(_spec)
sys.modules['evaluate_trex'] = ev
_spec.loader.exec_module(ev)

WD = 'http://www.wikidata.org/entity/'
WDT = 'http://www.wikidata.org/prop/direct/'

KB = {'Ada': 'Q7259', 'London': 'Q84', 'place of birth': 'P19',
      'Babbage': 'Q46633', 'employer': 'P108', 'Paris': 'Q90'}


class FakeExtractor(TripleExtractor, strategy_name='test-trex-extract'):
    """One triple per sentence, read from the sentence itself."""

    def __init__(self, strategy_name, fail_on=None, **kwargs):
        super().__init__(**kwargs)
        self.fail_on = fail_on

    def _extract(self, text, prompt=None):
        if self.fail_on and self.fail_on in text:
            raise RuntimeError('extractor exploded')
        if 'born' in text:
            return [Triple('Ada', 'place of birth', 'London' if 'London' in text else 'Paris')]
        if 'worked' in text:
            return [Triple('Ada', 'employer', 'Babbage'), Triple('Nobody', 'employer', 'Babbage')]
        return []


class FirstCandidate(Disambiguator, strategy_name='test-trex-first'):
    def __init__(self, strategy_name):
        pass

    def _disambiguate(self, mention, candidates, limit, *args, **kwargs):
        return [DisambiguationResult(candidate=c, score=1.0) for c in candidates[:limit]]


@pytest.fixture(autouse=True)
def fake_search(monkeypatch):
    def lookup(self, mention, limit=None):
        qid = KB.get(mention.term)
        return [Candidate(id=qid, iri=WD + qid, label=mention.term, description='d')] if qid else []
    monkeypatch.setattr(KBEL, 'candidates_lookup', lookup)


def triple(s, p, o, sentence=0, annotator='Simple-Aligner', literal=False):
    return {'sentence_id': sentence, 'annotator': annotator,
            'subject': {'uri': WD + s, 'surfaceform': s},
            'predicate': {'uri': WDT + p, 'surfaceform': None},
            'object': {'uri': o if literal else WD + o, 'surfaceform': o}}


def document(qid, sentences, triples):
    text, bounds = '', []
    for s in sentences:
        start = len(text) + (1 if text else 0)
        text = f'{text} {s}' if text else s
        bounds.append([start, len(text)])
    return {'docid': WD + qid, 'uri': WD + qid, 'title': qid, 'text': text,
            'sentences_boundaries': bounds, 'triples': triples, 'entities': []}


DOCS = [
    document('Q1', ['Ada was born in London.', 'Ada worked with Babbage.'], [
        triple('Q7259', 'P19', 'Q84'),
        triple('Q7259', 'P19', 'Q84', annotator='SPOAligner'),
        triple('Q7259', 'P108', 'Q46633', sentence=1),
        triple('Q7259', 'P569', '1815-12-10T00:00:00Z^^xsd:dateTime', literal=True),
    ]),
    document('Q2', ['Ada was born in Paris.'], [triple('Q7259', 'P19', 'Q84')]),
    document('Q3', ['Nothing here.'], []),
]


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / 'trex.json'
    path.write_text(json.dumps(DOCS))
    return path


def run(dataset, out, *extra):
    return ev.main(['run', '--dataset', str(dataset), '-o', str(out), '--kb', 'empty',
                    '--extractor', 'test-trex-extract', '--disambiguator', 'test-trex-first',
                    '--no-progress', '--no-summary', '--retry-backoff', '0',
                    '--env-file', '/dev/null', *extra])


def records(out):
    return [json.loads(line) for line in out.read_text().splitlines()]


def test_wikidata_id():
    assert ev.wikidata_id(WD + 'Q42') == 'Q42'
    assert ev.wikidata_id(WDT + 'P31') == 'P31'
    assert ev.wikidata_id('http://dbpedia.org/resource/Q42') == 'http://dbpedia.org/resource/Q42'


def test_gold_dedupes_and_drops_literals():
    assert set(ev.gold_triples(DOCS[0])) == {('Q7259', 'P19', 'Q84'), ('Q7259', 'P108', 'Q46633')}
    assert len(ev.gold_triples(DOCS[0], include_literals=True)) == 3
    assert set(ev.gold_triples(DOCS[0], annotators={'SPOAligner'})) == {('Q7259', 'P19', 'Q84')}
    assert set(ev.gold_triples(DOCS[0], max_sentences=1)) == {('Q7259', 'P19', 'Q84')}


def test_truncate_to_sentences():
    text, n = ev.truncate(DOCS[0], 1)
    assert text == 'Ada was born in London.' and n == 1
    assert ev.truncate(DOCS[0], None) == (DOCS[0]['text'], 2)


def test_prf():
    m = ev.prf({1, 2, 3, 4}, {1, 2, 5})
    assert (m['tp'], m['precision'], m['recall']) == (2, 0.5, 2 / 3)
    assert m['f1'] == pytest.approx(2 * 0.5 * (2 / 3) / (0.5 + 2 / 3))
    assert ev.prf(set(), set())['f1'] == 0.0


def test_micro_sums_counts():
    rows = [{'metrics': {'strict': ev.prf({1}, {1, 2})}},
            {'metrics': {'strict': ev.prf({3, 4}, {3})}}]
    m = ev.micro(rows, 'strict')
    assert (m['tp'], m['predicted'], m['gold']) == (2, 3, 3)


def test_parse_options():
    assert ev.parse_options(['a=6', 'b=x', 'c={"t": 0}', 'd=true']) == {
        'a': 6, 'b': 'x', 'c': {'t': 0}, 'd': True}
    with pytest.raises(SystemExit):
        ev.parse_options(['novalue'])


def args_for_llm(**kw):
    return ev.build_parser().parse_args(['run', '--env-file', '/dev/null', *kw.get('argv', [])])


def test_llm_plugins_default_to_openrouter(monkeypatch):
    monkeypatch.delenv('LLM_API_KEY', raising=False)
    monkeypatch.setenv('OPENROUTER_API_KEY', 'secret')
    opts, logged = ev.plugin_options('llm', {'model_params': {'max_tokens': 5}}, args_for_llm())
    assert opts['model_endpoint'] == ev.OPENROUTER_ENDPOINT
    assert opts['model_provider'] == 'openai' and opts['model_name'] == ev.DEFAULT_LLM_MODEL
    assert opts['model_params'] == {'temperature': 0.0, 'max_tokens': 5}
    assert opts['model_apikey'] == 'secret'
    assert logged['model_apikey'] == '<env:OPENROUTER_API_KEY>'


def test_llm_without_key_fails_early(monkeypatch):
    for name in ev.DEFAULT_LLM_KEY_ENVS:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(SystemExit, match='API key'):
        ev.plugin_options('llm', {}, args_for_llm())


def test_non_llm_options_untouched():
    assert ev.plugin_options('mrebel', {'num_beams': 3}, args_for_llm()) == (
        {'num_beams': 3}, {'num_beams': 3})


def test_run_writes_one_line_per_document(dataset, tmp_path):
    out = tmp_path / 'run.jsonl'
    assert run(dataset, out) == 0
    q1, q2, q3 = records(out)
    assert [r['docid'] for r in (q1, q2, q3)] == ['Q1', 'Q2', 'Q3']
    assert all(r['status'] == 'ok' for r in (q1, q2, q3))

    assert (q1['precision'], q1['recall'], q1['f1']) == (1.0, 1.0, 1.0)
    assert {(s['subject'], s['property'], s['value']) for s in q1['statements']} == {
        ('Q7259', 'P19', 'Q84'), ('Q7259', 'P108', 'Q46633')}
    nobody = next(t for t in q1['extracted_triples'] if t['subject'] == 'Nobody')
    assert nobody['linked']['subject'] == {'id': None, 'reason': 'no_candidates'}
    assert nobody['statement'] is None and nobody['correct'] is None
    first = q1['extracted_triples'][0]
    assert first['statement'] == ['Q7259', 'P19', 'Q84'] and first['correct'] is True
    assert first['linked']['object']['label'] == 'London'
    assert all(g['found'] for g in q1['gold'])

    assert q2['metrics']['strict']['tp'] == 0 and q2['metrics']['strict']['predicted'] == 1
    assert q2['metrics']['entities']['tp'] == 1
    assert q2['gold'][0]['found'] is False

    t = q1['timing']
    assert t['extraction']['calls'] == 2
    assert t['search']['calls'] == 6
    assert t['disambiguation']['calls'] == 6
    assert t['total_s'] >= t['extraction']['active_s'] > 0
    assert set(q1['trace']) == {'extractions', 'searches', 'disambiguations'}
    assert q1['trace']['disambiguations'][0]['results'][0]['score'] == 1.0
    assert q1['config_hash'] and q1['attempts'] == 1

    config = json.loads((tmp_path / 'run.jsonl.config.json').read_text())
    assert config['config_hash'] == q1['config_hash']
    assert config['fingerprint']['extractor'] == 'test-trex-extract'
    assert not (tmp_path / 'run.jsonl.lock').exists()


def test_trace_none_and_selection(dataset, tmp_path):
    out = tmp_path / 'run.jsonl'
    assert run(dataset, out, '--trace', 'none', '--docids', 'Q2,Q3', '--limit', '1') == 0
    [r] = records(out)
    assert r['docid'] == 'Q2' and 'trace' not in r


def test_resume_skips_finished_documents(dataset, tmp_path):
    out = tmp_path / 'run.jsonl'
    run(dataset, out, '--limit', '1')
    run(dataset, out)
    assert [r['docid'] for r in records(out)] == ['Q1', 'Q2', 'Q3']


def test_resume_survives_a_truncated_last_line(dataset, tmp_path):
    out = tmp_path / 'run.jsonl'
    run(dataset, out, '--limit', '1')
    with out.open('a') as f:
        f.write('{"docid": "Q2", "stat')
    run(dataset, out)
    assert [r['docid'] for r in ev.read_jsonl(out)] == ['Q1', 'Q2', 'Q3']


def test_different_configuration_is_refused(dataset, tmp_path):
    out = tmp_path / 'run.jsonl'
    run(dataset, out, '--limit', '1')
    with pytest.raises(SystemExit, match='search_limit'):
        run(dataset, out, '--search-limit', '3')
    assert run(dataset, out, '--search-limit', '3', '--force') == 0


def test_overwrite_backs_up(dataset, tmp_path):
    out = tmp_path / 'run.jsonl'
    run(dataset, out, '--limit', '1')
    run(dataset, out, '--search-limit', '3', '--overwrite', '--limit', '1')
    backups = sorted(p.name for p in tmp_path.glob('run.jsonl*.bak'))
    assert len(backups) == 2
    assert len(records(out)) == 1


def test_failures_are_recorded_retried_and_resumable(dataset, tmp_path):
    out = tmp_path / 'run.jsonl'
    assert run(dataset, out, '-E', 'fail_on=Paris', '--retries', '1') == 0
    by_id = {r['docid']: r for r in records(out)}
    q2 = by_id['Q2']
    assert q2['status'] == 'error' and q2['attempts'] == 2
    assert q2['error']['message'] == 'extractor exploded'
    assert 'Traceback' in q2['error']['traceback']
    assert by_id['Q1']['status'] == by_id['Q3']['status'] == 'ok'

    run(dataset, out, '-E', 'fail_on=Paris')
    assert len(records(out)) == 3
    run(dataset, out, '-E', 'fail_on=Paris', '--retry-failed', '--retries', '0')
    assert [r['docid'] for r in records(out)] == ['Q1', 'Q2', 'Q3', 'Q2']


def test_consecutive_errors_abort(dataset, tmp_path):
    out = tmp_path / 'run.jsonl'
    code = run(dataset, out, '-E', 'fail_on=.', '--retries', '0', '--max-consecutive-errors', '2')
    assert code == 1
    assert [r['status'] for r in records(out)] == ['error', 'error']


def test_timeout(dataset, tmp_path, monkeypatch):
    import time as _time
    monkeypatch.setattr(FakeExtractor, '_extract', lambda self, text, prompt=None:
                        _time.sleep(0.5) or [])
    out = tmp_path / 'run.jsonl'
    run(dataset, out, '--timeout', '0.1', '--retries', '0', '--limit', '1')
    [r] = records(out)
    assert r['status'] == 'timeout' and r['error']['type'] == 'TimeoutError'


def test_unknown_plugin(dataset, tmp_path):
    with pytest.raises(SystemExit, match='Unknown extractor'):
        ev.main(['run', '--dataset', str(dataset), '-o', str(tmp_path / 'x.jsonl'),
                 '--kb', 'empty', '--extractor', 'nope', '--disambiguator', 'test-trex-first',
                 '--env-file', '/dev/null'])


def test_dry_run_writes_nothing(dataset, tmp_path, capsys):
    out = tmp_path / 'run.jsonl'
    assert run(dataset, out, '--dry-run') == 0
    assert json.loads(capsys.readouterr().out)['to_run'] == ['Q1', 'Q2', 'Q3']
    assert not out.exists()


def test_summary_compares_common_documents(dataset, tmp_path, capsys):
    a, b = tmp_path / 'a.jsonl', tmp_path / 'b.jsonl'
    run(dataset, a)
    run(dataset, b, '-E', 'fail_on=London', '--retries', '0')
    capsys.readouterr()
    assert ev.main(['summary', str(a), str(b), '--format', 'json']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['common_documents'] == 2
    sa, sb = report['runs']
    assert sa['strict']['micro']['gold'] == sb['strict']['micro']['gold'] == 1
    assert sa['ok'] == 2 and sb['error'] == 0

    ev.main(['summary', str(a), str(b), '--all-docs'])
    table = capsys.readouterr().out
    assert 'strict micro P / R / F1' in table and 'a.jsonl' in table


def test_show(dataset, tmp_path, capsys):
    out = tmp_path / 'run.jsonl'
    run(dataset, out)
    capsys.readouterr()
    ev.main(['show', str(out)])
    assert capsys.readouterr().out.count('ok') == 3
    ev.main(['show', str(out), 'Q2'])
    text = capsys.readouterr().out
    assert 'XX (Ada | place of birth | Paris)' in text and 'gold triples missed (1/1)' in text
    ev.main(['show', str(out), 'Q1', '--json'])
    assert json.loads(capsys.readouterr().out)['docid'] == 'Q1'


def test_kggen_defaults_to_openrouter_through_litellm(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'secret')
    opts, logged = ev.plugin_options('kggen-cluster', {'cluster_edges': True}, args_for_llm())
    assert opts == {'model': f'openrouter/{ev.DEFAULT_LLM_MODEL}', 'temperature': 0.0,
                    'cluster_edges': True, 'api_key': 'secret'}
    assert logged['api_key'] == '<env:OPENROUTER_API_KEY>'
    other = ev.build_parser().parse_args(['run', '--llm-endpoint', 'http://localhost:11434/v1'])
    opts, _ = ev.plugin_options('kggen', {}, other)
    assert opts['model'] == f'openai/{ev.DEFAULT_LLM_MODEL}'
    assert opts['api_base'] == 'http://localhost:11434/v1'
