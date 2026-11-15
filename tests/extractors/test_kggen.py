# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

"""KGGen extractors with a scripted KGGen (no LLM calls)."""

import asyncio
import os
import threading
import time

import pytest

pytest.importorskip('kg_gen')

from kg_gen import KGGen  # noqa: E402
from kg_gen.models import Graph  # noqa: E402

from kbel.core.triple import Triple  # noqa: E402
from kbel.extractors import TripleExtractor  # noqa: E402
from kbel.extractors import kggen  # noqa: E402,F401

RELATIONS = {
    'Turing was born in London.': {('Turing', 'was born in', 'London')},
    'Alan Turing died in Wilmslow.': {('Alan Turing', 'died in', 'Wilmslow')},
    'Alan Turing  invented the  Turing machine.': {
        ('Alan Turing', 'invented ', ' the  Turing machine'), ('Alan Turing', 'is', 'Alan Turing')},
}


class ScriptedKG:
    def __init__(self, delay=0.0, entity_clusters=None, edge_clusters=None):
        self.delay = delay
        self.entity_clusters = entity_clusters or {}
        self.edge_clusters = edge_clusters or {}
        self.texts, self.threads, self.clustered = [], set(), []

    def generate(self, input_data):
        self.texts.append(input_data)
        self.threads.add(threading.get_ident())
        time.sleep(self.delay)
        relations = RELATIONS.get(input_data, set())
        return Graph(entities={x for r in relations for x in (r[0], r[2])},
                     relations=relations, edges={r[1] for r in relations})

    aggregate = KGGen.aggregate

    def cluster(self, graph, context=''):
        self.clustered.append((graph, context))
        return Graph(entities=graph.entities, relations=graph.relations, edges=graph.edges,
                     entity_clusters=self.entity_clusters, edge_clusters=self.edge_clusters)


def test_kggen_cleans_and_dedupes():
    e = TripleExtractor('kggen', kg=ScriptedKG())
    assert e.extract('Alan Turing  invented the  Turing machine.') == [
        Triple('Alan Turing', 'invented', 'the Turing machine')]


def test_kggen_batch_runs_texts_concurrently():
    kg = ScriptedKG(delay=0.2)
    e = TripleExtractor('kggen', kg=kg)
    texts = list(RELATIONS)[:2]
    start = time.perf_counter()
    result = asyncio.run(e.aextract_batch(texts))
    assert time.perf_counter() - start < 0.35
    assert result == [[Triple('Turing', 'was born in', 'London')],
                      [Triple('Alan Turing', 'died in', 'Wilmslow')]]
    assert e.batch_size == 8


def test_cluster_maps_entities_across_the_batch_but_not_edges_by_default():
    kg = ScriptedKG(entity_clusters={'Alan Turing': {'Alan Turing', 'Turing'}},
                    edge_clusters={'birth and death': {'was born in', 'died in'}})
    e = TripleExtractor('kggen-cluster', kg=kg, context='biographies')
    result = e.extract_batch(['Turing was born in London.', 'Alan Turing died in Wilmslow.'])
    assert result == [[Triple('Alan Turing', 'was born in', 'London')],
                      [Triple('Alan Turing', 'died in', 'Wilmslow')]]
    [(merged, context)] = kg.clustered
    assert len(merged.relations) == 2 and context == 'biographies'


def test_cluster_edges_option():
    kg = ScriptedKG(edge_clusters={'birth and death': {'was born in', 'died in'}})
    e = TripleExtractor('kggen-cluster', kg=kg, cluster_edges=True)
    assert e.extract('Turing was born in London.') == [
        Triple('Turing', 'birth and death', 'London')]


def test_cluster_drops_self_loops_created_by_merging():
    kg = ScriptedKG(entity_clusters={'London': {'London', 'Turing'}})
    e = TripleExtractor('kggen-cluster', kg=kg)
    assert e.extract('Turing was born in London.') == []


def test_cluster_skips_the_llm_when_there_is_nothing_to_cluster():
    kg = ScriptedKG()
    e = TripleExtractor('kggen-cluster', kg=kg)
    assert e.extract_batch(['Nothing here.', 'Nor here.']) == [[], []]
    assert kg.clustered == []


def test_construct_sends_batches_to_kggen_cluster(monkeypatch):
    from kbel.core.candidate import Candidate
    from kbel.core.kbel import KBEL
    from kbel.disambiguators import naive  # noqa: F401

    monkeypatch.setattr(KBEL, 'candidates_lookup', lambda self, m, limit=None: [
        Candidate(id=m.term, iri=f'http://www.wikidata.org/entity/{m.term.replace(" ", "_")}',
                  label=m.term)])
    kg = ScriptedKG(entity_clusters={'Alan Turing': {'Turing', 'Alan Turing'}})
    e = TripleExtractor('kggen-cluster', kg=kg, batch_size=2)
    kbel = KBEL('empty', disambiguator='naive', extractor=e)
    statements = list(kbel.construct('Turing was born in London. Alan Turing died in Wilmslow.'))
    assert {str(s.subject.iri.content).rsplit('/', 1)[1] for s in statements} == {'Alan_Turing'}
    assert len(kg.clustered) == 1


def test_unknown_option_is_rejected():
    with pytest.raises(TypeError):
        TripleExtractor('kggen', kg=ScriptedKG(), cluster_edges=True)


@pytest.mark.integration
def test_real_kggen_through_openrouter():
    from dotenv import load_dotenv
    load_dotenv()
    key = os.environ.get('OPENROUTER_API_KEY') or os.environ.get('LLM_API_KEY')
    if not key:
        pytest.skip('no OpenRouter key')
    e = TripleExtractor('kggen', api_key=key)
    triples = e.extract('Python was created by Guido van Rossum.')
    assert any('Guido' in t.subject or 'Guido' in t.object for t in triples)
