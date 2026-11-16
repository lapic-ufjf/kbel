# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

"""scripts/evaluate_trex.py, offline: fake plugins and a tiny T-REx file."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from kif_lib import TimeDatatype

from kbel.core.candidate import Candidate
from kbel.core.kbel import KBEL
from kbel.core.results import DisambiguationResult
from kbel.core.triple import Triple
from kbel.coreference import CoreferenceResolver
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
      'Babbage': 'Q46633', 'employer': 'P108', 'Paris': 'Q90', 'date of birth': 'P569',
      'date of death': 'P570'}


class FakeExtractor(TripleExtractor, strategy_name='test-trex-extract'):
    """One triple per sentence, read from the sentence itself."""

    def __init__(self, strategy_name, fail_on=None, **kwargs):
        super().__init__(**kwargs)
        self.fail_on = fail_on

    def _extract(self, text, prompt=None):
        if self.fail_on and self.fail_on in text:
            raise RuntimeError('extractor exploded')
        if 'On ' in text:
            return [Triple('Ada', 'date of birth', '10 December 1815'),
                    Triple('Ada', 'date of death', 'unknown')]
        if 'born' in text:
            return [Triple('Ada', 'place of birth', 'London' if 'London' in text else 'Paris')]
        if 'worked' in text:
            who = 'She' if text.startswith('She') else 'Ada'
            return [Triple(who, 'employer', 'Babbage'), Triple('Nobody', 'employer', 'Babbage')]
        return []


class FirstCandidate(Disambiguator, strategy_name='test-trex-first'):
    def __init__(self, strategy_name):
        pass

    def _disambiguate(self, mention, candidates, limit, *args, **kwargs):
        return [DisambiguationResult(candidate=c, score=1.0) for c in candidates[:limit]]


class SheIsAda(CoreferenceResolver, strategy_name='test-trex-coref'):
    def __init__(self, strategy_name, **kwargs):
        super().__init__(**kwargs)

    def _clusters(self, texts):
        return [[[(0, 3), (t.index('She'), t.index('She') + 3)]] if 'She' in t else []
                for t in texts]


@pytest.fixture(autouse=True)
def fake_search(monkeypatch):
    def lookup(self, mention, limit=None):
        qid = KB.get(mention.term)
        datatype = TimeDatatype() if mention.term in ('date of birth', 'date of death') else None
        return [Candidate(id=qid, iri=WD + qid, label=mention.term, description='d',
                          datatype=datatype)] if qid else []
    monkeypatch.setattr(KBEL, 'candidates_lookup', lookup)


def triple(s, p, o, sentence=0, annotator='Simple-Aligner', literal=False):
    return {'sentence_id': sentence, 'annotator': annotator,
            'subject': {'uri': WD + s, 'surfaceform': s},
            'predicate': {'uri': WDT + p, 'surfaceform': None},
            'object': {'uri': o if literal else WD + o, 'surfaceform': o}}


ENTITIES = [{'surfaceform': n, 'uri': WD + q}
            for n, q in [('Ada', 'Q7259'), ('London', 'Q84'), ('Babbage', 'Q46633')]]


def document(qid, sentences, triples):
    text, bounds = '', []
    for s in sentences:
        start = len(text) + (1 if text else 0)
        text = f'{text} {s}' if text else s
        bounds.append([start, len(text)])
    return {'docid': WD + qid, 'uri': WD + qid, 'title': qid, 'text': text,
            'sentences_boundaries': bounds, 'triples': triples, 'entities': ENTITIES}


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


@pytest.mark.parametrize('iri,expected', [
    ('http://www.wikidata.org/entity/Q42', 'Q42'),
    ('https://www.wikidata.org/entity/Q42', 'Q42'),
    ('https://www.wikidata.org/wiki/Q42', 'Q42'),
    ('http://www.wikidata.org/prop/direct/P31', 'P31'),
    ('http://www.wikidata.org/entity/P31', 'P31'),
    ('https://www.wikidata.org/prop/P31', 'P31'),
    ('Q42', 'Q42'),
])
def test_ids_are_compared_without_the_iri_scheme_or_namespace(iri, expected):
    assert ev.wikidata_id(iri) == expected


def test_statements_match_gold_by_q_and_p_ids_only():
    from kif_lib import Item, Property, Statement, ValueSnak

    gold = ev.gold_triples(DOCS[0])
    assert ('Q7259', 'P19', 'Q84') in gold
    predicted = Statement(Item('https://www.wikidata.org/entity/Q7259'),
                          ValueSnak(Property('http://www.wikidata.org/entity/P19'),
                                    Item('http://www.wikidata.org/wiki/Q84')))
    ids = ev._statement_ids(predicted)
    assert ids == ('Q7259', 'P19', 'Q84')
    assert ev.evaluate({ids}, set(gold))['strict']['tp'] == 1
    other_namespace = Statement(Item('http://dbpedia.org/resource/Q7259'),
                                ValueSnak(Property(WDT + 'P19'), Item(WD + 'Q84')))
    assert ev._statement_ids(other_namespace)[0] == 'http://dbpedia.org/resource/Q7259'
    assert ev.evaluate({ev._statement_ids(other_namespace)}, set(gold))['strict']['tp'] == 0


def test_records_store_only_ids(dataset, tmp_path):
    out = tmp_path / 'run.jsonl'
    run(dataset, out)
    for record in records(out):
        for s in record['statements']:
            assert s['subject'][0] == 'Q' and s['property'][0] == 'P' and s['value'][0] == 'Q'
        for g in record['gold']:
            assert g['subject'][0] == 'Q' and g['property'][0] == 'P' and g['value'][0] == 'Q'


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
    assert set(q1['trace']) == {'coreferences', 'extractions', 'searches', 'disambiguations'}
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


def linked(s=None, p=None, o=None):
    def part(i):
        return {'id': i, 'label': None} if i else {'id': None, 'reason': 'no_candidates'}
    return {'subject': part(s), 'predicate': part(p), 'object': part(o)}


def triple_(subject, predicate, obj, s=None, p=None, o=None):
    ids = (s, p, o)
    return {'subject': subject, 'predicate': predicate, 'object': obj,
            'linked': linked(s, p, o), 'statement': list(ids) if all(ids) else None}


def test_diagnose_locates_each_error():
    gold = {('Q1', 'P19', 'Q2'): {}, ('Q1', 'P27', 'Q3'): {}, ('Q4', 'P17', 'Q3'): {},
            ('Q5', 'P31', 'Q6'): {}, ('Q7', 'P50', 'Q8'): {}}
    aliases = {'ada': {'Q1'}, 'london': {'Q2'}, 'england': {'Q3'}, 'oxford': {'Q4'},
               'bob': {'Q5'}, 'human': {'Q6'}, 'book': {'Q7'}, 'carl': {'Q8'}}
    candidates = {('Ada', 'item'): ['Q1'], ('London', 'item'): ['Q99', 'Q2'],
                  ('England', 'item'): ['Q98'], ('born in', 'property'): ['P19'],
                  ('citizen of', 'property'): ['P1', 'P27'], ('lives in', 'property'): ['P551']}
    extracted = [
        triple_('Ada', 'born in', 'London', 'Q1', 'P19', 'Q2'),
        triple_('The Ada', 'citizen of', 'England', 'Q1', 'P1', 'Q3'),
        triple_('Ada', 'lives in', "England's", 'Q1', 'P551', 'Q3'),
        triple_('Ada', 'born in', 'London', 'Q1', 'P19', 'Q99'),
        triple_('Ada', 'visited', 'England', 'Q1', 'P19', 'Q98'),
        triple_('England', 'contains', 'Oxford', 'Q3', 'P150', 'Q4'),
        triple_('Ada', 'met', 'Bob', 'Q1', 'P1', 'Q5'),
        triple_('Ada', 'likes', 'tea', 'Q1', 'P1', 'Q9'),
        triple_('Human', 'is', 'Ada', 'Q6', 'P1', 'Q1'),
    ]
    causes = ev.diagnose(extracted, gold, aliases, lambda t, kind: candidates.get((t, kind)))
    assert [t['stage'] for t in extracted] == [
        'correct', 'property_disambiguation', 'property_search', 'entity_disambiguation',
        'entity_search', 'inverted_direction', 'relation_not_in_gold', 'extraction',
        'relation_not_in_gold']
    assert extracted[7]['diagnosis']['object'] == 'extraction'
    assert extracted[0]['diagnosis']['gold_subject'] == ['Q1']
    assert causes == {('Q1', 'P19', 'Q2'): 'found', ('Q1', 'P27', 'Q3'): 'property_disambiguation',
                      ('Q4', 'P17', 'Q3'): 'inverted_direction',
                      ('Q5', 'P31', 'Q6'): 'pair_not_extracted', ('Q7', 'P50', 'Q8'): 'not_extracted'}


def test_records_carry_the_diagnosis(dataset, tmp_path):
    out = tmp_path / 'run.jsonl'
    run(dataset, out)
    q1, q2, _ = records(out)
    assert {t['stage'] for t in q1['extracted_triples'] if t['subject'] == 'Ada'} == {'correct'}
    nobody = next(t for t in q1['extracted_triples'] if t['subject'] == 'Nobody')
    assert nobody['stage'] == 'extraction'
    assert [g['cause'] for g in q1['gold']] == ['found', 'found']
    assert q2['extracted_triples'][0]['stage'] == 'extraction'
    assert q2['extracted_triples'][0]['diagnosis']['object'] == 'extraction'
    assert q2['gold'][0]['cause'] == 'not_extracted'


@pytest.mark.parametrize('term,reason', [
    ('It', 'pronoun'), ('they', 'pronoun'), ('1st of April 2006', 'literal'), ('twelve', 'literal'),
    ('June 2010', 'literal'), ('free-to-air television network', 'unannotated_span')])
def test_extraction_reason(term, reason):
    assert ev.extraction_reason(term) == reason


def test_normalize_ignores_articles_possessives_and_dash_spacing():
    assert ev.normalize('IBM Research – Brazil') == ev.normalize('IBM Research–Brazil')
    assert ev.normalize("The  Skai Group's") == 'skai group'


def test_extraction_errors_carry_the_reason():
    extracted = [triple_('It', 'founded in', '2006', 'Q77', 'P571', 'Q9'),
                 triple_('Ada', 'is', 'a famous person', 'Q1', 'P31', 'Q5')]
    ev.diagnose(extracted, {}, {'ada': {'Q1'}}, lambda t, kind: [])
    assert [t['diagnosis']['extraction'] for t in extracted] == ['pronoun', 'unannotated_span']


def test_diagnose_accepts_unannotated_spans_linked_to_annotated_entities():
    gold = {('Q1', 'P17', 'Q2'): {}, ('Q1', 'P131', 'Q2'): {}, ('Q3', 'P17', 'Q2'): {}}
    aliases = {'portland': {'Q1'}, 'united states': {'Q2'}, 'oregon': {'Q3'}}
    extracted = [
        triple_('Portland, Oregon', 'country', 'United States', 'Q1', 'P17', 'Q2'),
        triple_('It', 'is in', 'the US', 'Q3', 'P131', 'Q2'),
        triple_('It', 'is in', 'the US', 'Q50', 'P131', 'Q2'),
    ]
    causes = ev.diagnose(extracted, gold, aliases, lambda t, kind: ['P131'])
    assert [t['stage'] for t in extracted] == ['correct', 'property_search', 'extraction']
    assert extracted[0]['diagnosis']['subject'] == 'correct'
    assert extracted[1]['diagnosis']['gold_subject'] == ['Q3']
    assert extracted[2]['diagnosis']['extraction'] == 'pronoun'
    assert causes == {('Q1', 'P17', 'Q2'): 'found', ('Q1', 'P131', 'Q2'): 'relation_not_extracted',
                      ('Q3', 'P17', 'Q2'): 'property_search'}


def test_diagnose_blames_the_link_when_a_name_has_several_annotated_entities():
    gold = {('Q1', 'P17', 'Q41'): {}}
    aliases = {'skai': {'Q1'}, 'greek': {'Q41', 'Q9129'}}
    extracted = [triple_('Skai', 'country', 'Greek', 'Q1', 'P17', 'Q9129')]
    causes = ev.diagnose(extracted, gold, aliases, lambda t, kind: [])
    assert extracted[0]['stage'] == 'relation_not_in_gold'
    assert causes == {('Q1', 'P17', 'Q41'): 'entity_disambiguation'}
    assert set(causes.values()) <= set(ev.GOLD_CAUSES)


def test_rediagnose_recomputes_from_the_trace(dataset, tmp_path):
    out = tmp_path / 'run.jsonl'
    run(dataset, out)
    original = records(out)
    tampered = [{**r, 'extracted_triples': [{**t, 'stage': 'x'} for t in r['extracted_triples']],
                 'gold': [{**g, 'cause': 'x'} for g in r['gold']]} for r in original]
    out.write_text(''.join(json.dumps(r) + '\n' for r in tampered))
    assert ev.main(['rediagnose', str(out), '--dataset', str(dataset)]) == 0
    redone = records(out)
    assert [[t['stage'] for t in r['extracted_triples']] for r in redone] == \
        [[t['stage'] for t in r['extracted_triples']] for r in original]
    assert [[g['cause'] for g in r['gold']] for r in redone] == \
        [[g['cause'] for g in r['gold']] for r in original]
    assert len(list(tmp_path.glob('run.jsonl.*.bak'))) == 1


@pytest.fixture
def coref_dataset(tmp_path):
    doc = document('Q9', ['Ada was born in London.', 'She worked with Babbage.'], [
        triple('Q7259', 'P19', 'Q84'), triple('Q7259', 'P108', 'Q46633', sentence=1)])
    path = tmp_path / 'coref.json'
    path.write_text(json.dumps([doc]))
    return path


def test_coreference_is_resolved_before_the_pipeline_and_recorded(coref_dataset, tmp_path):
    plain, resolved = tmp_path / 'plain.jsonl', tmp_path / 'resolved.jsonl'
    run(coref_dataset, plain)
    run(coref_dataset, resolved, '--coreference', 'test-trex-coref')
    [before], [after] = records(plain), records(resolved)
    assert before['metrics']['strict']['tp'] == 1
    assert after['metrics']['strict']['tp'] == 2
    assert 'coreference' not in before
    assert after['coreference'] == {
        'resolved_text': 'Ada was born in London. Ada worked with Babbage.', 'changed': True}
    assert after['timing']['coreference']['calls'] == 1
    assert before['timing']['coreference']['calls'] == 0
    assert sorted(x for e in after['trace']['extractions'] for x in e['sentences']) == [
        'Ada was born in London.', 'Ada worked with Babbage.']
    assert after['trace']['coreferences'][0]['changed'] is True
    assert 'She' not in {t['subject'] for t in after['extracted_triples']}


def test_coreference_is_part_of_the_configuration(coref_dataset, tmp_path):
    out = tmp_path / 'run.jsonl'
    run(coref_dataset, out, '--coreference', 'test-trex-coref', '-C', 'replace=all')
    config = json.loads((tmp_path / 'run.jsonl.config.json').read_text())['fingerprint']
    assert config['coreference'] == 'test-trex-coref'
    assert config['coreference_options'] == {'replace': 'all'}
    with pytest.raises(SystemExit, match='another configuration'):
        run(coref_dataset, out)
    with pytest.raises(SystemExit, match='Unknown coreference'):
        run(coref_dataset, tmp_path / 'other.jsonl', '--coreference', 'nope')


def test_summary_and_default_names_mention_the_coreference(coref_dataset, tmp_path, capsys):
    out = tmp_path / 'run.jsonl'
    run(coref_dataset, out, '--coreference', 'test-trex-coref')
    ev.main(['summary', str(out)])
    text = capsys.readouterr().out
    assert 'empty/test-trex-coref/test-trex-extract/test-trex-first' in text
    assert 'coreference busy s/doc' in text
    args = ev.build_parser().parse_args(['run', '--coreference', 'fastcoref'])
    assert 'fastcoref' in ev.default_output(args).name


@pytest.fixture
def literal_dataset(tmp_path):
    doc = document('Q8', ['On 10 December 1815 Ada was born.'], [
        triple('Q7259', 'P569', '1815-12-10T00:00:00Z^^http://www.w3.org/2001/XMLSchema#dateTime',
               literal=True)])
    path = tmp_path / 'literals.json'
    path.write_text(json.dumps([doc]))
    return path


def test_literal_statements_are_left_out_of_the_entity_only_metrics(literal_dataset, tmp_path):
    out = tmp_path / 'run.jsonl'
    run(literal_dataset, out)
    [record] = records(out)
    assert record['metrics']['strict'] == {
        'tp': 0, 'predicted': 0, 'gold': 0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    by_predicate = {t['predicate']: t for t in record['extracted_triples']}
    born = by_predicate['date of birth']
    assert born['object_kind'] == 'literal' and born['correct'] is None
    assert born['statement'] == ['Q7259', 'P569', 'time:1815-12-10']
    assert born['stage'] == 'literal'
    assert by_predicate['date of death']['object_kind'] == 'unknown'
    assert by_predicate['date of death']['statement'] == ['Q7259', 'P570', 'unknown']
    assert {s['value'] for s in record['statements']} == {'time:1815-12-10', 'unknown'}
    assert all(s['correct'] is None for s in record['statements'])


def test_literal_statements_are_compared_by_value_when_included(literal_dataset, tmp_path):
    out = tmp_path / 'run.jsonl'
    run(literal_dataset, out, '--include-literals')
    [record] = records(out)
    strict = record['metrics']['strict']
    assert (strict['tp'], strict['predicted'], strict['gold']) == (1, 2, 1)
    born = next(t for t in record['extracted_triples'] if t['predicate'] == 'date of birth')
    assert born['correct'] is True and born['stage'] == 'correct'
    assert record['gold'][0]['found'] is True and record['gold'][0]['value'] == 'time:1815-12-10'
    assert record['metrics']['entities']['predicted'] == 1


def test_value_keys_compare_entities_by_id_and_literals_by_value():
    from kif_lib import Item, Quantity, String, Text, Time

    assert ev.value_key(Item(WD + 'Q42')) == 'Q42'
    assert ev.value_key(Item('https://www.wikidata.org/wiki/Q42')) == 'Q42'
    assert ev.value_key(Quantity('3.0')) == 'quantity:3'
    assert ev.value_key(Quantity(5, Item(WD + 'Q11570'))) == 'quantity:5|Q11570'
    assert ev.value_key(String('x')) == 'string:x' and ev.value_key(Text('x', 'pt')) == 'text:x'
    assert ev.value_key(Time('1815-12-10')) == 'time:1815-12-10'
    assert ev.gold_value_key('+3.50^^http://www.w3.org/2001/XMLSchema#decimal') == 'quantity:3.5'
    assert ev.gold_value_key(WD + 'Q42') == 'Q42'
    assert ev.is_entity_id('Q42') and ev.is_entity_id('P31')
    assert not ev.is_entity_id('time:1815-12-10') and not ev.is_entity_id('unknown')
