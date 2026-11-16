#!/usr/bin/env python3
# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

"""Evaluate KBEL on T-REx documents (``re-nlg_*.json``).

Every selected document goes through ``KBEL.aconstruct``: the pipeline that
splits the text, extracts triples, searches candidates and disambiguates
them. The statements it yields are compared with the document's T-REx
triples (Wikidata ids). Each document becomes one line of a JSONL file with
the extracted triples (and how each part was linked), precision, recall and
F1, the time spent in every stage and the total time, plus, optionally, the
full trace of every extraction, search and disambiguation call.

Runs are safe to interrupt: every document is written (and fsynced) as soon
as it finishes, and running the same command again resumes where it
stopped. The run configuration is kept next to the output, and resuming
with a different configuration is refused.

Subcommands::

    run         evaluate a configuration        (scripts/evaluate_trex.py run --help)
    grid        run every combination           (scripts/evaluate_trex.py grid --extractors a,b --disambiguators c,d)
    summary     aggregate and compare runs      (scripts/evaluate_trex.py summary a.jsonl b.jsonl)
    show        inspect one document of a run   (scripts/evaluate_trex.py show a.jsonl Q33199)
    rediagnose  recompute the error diagnosis   (scripts/evaluate_trex.py rediagnose a.jsonl)

Examples::

    # 20 documents, Wikidata search, mREBEL, naive disambiguation
    python scripts/evaluate_trex.py run --limit 20 --extractor mrebel --disambiguator naive

    # LLM extraction and disambiguation through OpenRouter (OPENROUTER_API_KEY)
    python scripts/evaluate_trex.py run --limit 20 --extractor llm --disambiguator llm \\
        --llm-model openai/gpt-4o-mini

    # plugin options are KEY=VALUE (values parsed as JSON when possible)
    python scripts/evaluate_trex.py run --extractor mrebel -E num_beams=6 -E batch_size=4 \\
        --disambiguator sim -D model_name=Snowflake/snowflake-arctic-embed-xs -D context_weight=1
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime as dt
import hashlib
import importlib
import json
import logging
import os
import platform
import random
import re
import signal
import statistics
import subprocess
import sys
import time
import traceback
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / 'evaluation' / 'dataset' / 're-nlg_0-10000.json'
DEFAULT_RUNS_DIR = ROOT / 'evaluation' / 'runs'

OPENROUTER_ENDPOINT = 'https://openrouter.ai/api/v1'
DEFAULT_LLM_MODEL = 'openai/gpt-4o-mini'
DEFAULT_LLM_KEY_ENVS = ('OPENROUTER_API_KEY', 'LLM_API_KEY')

EXTRACTOR_MODULES = {
    'openie': 'kbel.extractors.openie',
    'mrebel': 'kbel.extractors.mrebel',
    'llm': 'kbel.extractors.llm',
    'relik': 'kbel.extractors.relik',
    'relik-cie': 'kbel.extractors.relik',
    'kggen': 'kbel.extractors.kggen',
    'kggen-cluster': 'kbel.extractors.kggen',
}

LANGCHAIN_LLM_PLUGINS = {'llm'}
LITELLM_PLUGINS = {'kggen', 'kggen-cluster'}
COREFERENCE_MODULES = {
    'fastcoref': 'kbel.coreference.fastcoref',
}
DISAMBIGUATOR_MODULES = {
    'naive': 'kbel.disambiguators.naive',
    'sim': 'kbel.disambiguators.similarity',
    'llm': 'kbel.disambiguators.llm',
}

SCHEMA_VERSION = 1
LOG = logging.getLogger('evaluate_trex')

_WIKIDATA_ID = re.compile(r'([QP]\d+)$')


def wikidata_id(iri: str | None) -> str | None:
    """`http://www.wikidata.org/entity/Q42` (or `.../prop/direct/P31`) -> id.

    IRIs outside Wikidata are returned unchanged, so they never match.
    """
    if iri is None:
        return None
    m = _WIKIDATA_ID.search(iri)
    return m.group(1) if m and 'wikidata.org' in iri else iri


_ENTITY_ID = re.compile(r'^[QPL]\d+$')
_XSD = 'http://www.w3.org/2001/XMLSchema#'


def is_entity_id(key: str | None) -> bool:
    """Whether a statement value is a Wikidata entity id (not a literal or a marker)."""
    return bool(key and _ENTITY_ID.match(key))


def value_key(value: Any) -> str:
    """The key that compares a statement value with the gold.

    Entities are their Wikidata id (``Q42``, ``P31``), so IRIs are never
    compared. Literals are ``time:1815-12-10``, ``quantity:3000000``
    (``quantity:5|Q11570`` with a unit), ``string:...``, ``text:...``,
    ``iri:...``. A statement with a some-value / no-value snak has the value
    ``unknown`` / ``none`` (see `_statement_ids`).
    """
    from kif_lib import IRI, Entity, ExternalId, Quantity, String, Text, Time

    if isinstance(value, Entity):
        return wikidata_id(value.iri.content)
    if isinstance(value, Time):
        return f'time:{value.time.date().isoformat()}'
    if isinstance(value, Quantity):
        amount = format(value.amount.normalize(), 'f')
        unit = f'|{wikidata_id(value.unit.iri.content)}' if value.unit is not None else ''
        return f'quantity:{amount}{unit}'
    if isinstance(value, Text):
        return f'text:{value.content}'
    if isinstance(value, (String, ExternalId)):
        return f'string:{value.content}'
    if isinstance(value, IRI):
        return f'iri:{value.content}'
    return f'other:{value}'


def gold_value_key(uri: str) -> str:
    """The `value_key` of a gold object: an entity IRI or a typed literal."""
    if 'wikidata.org/entity/' in uri:
        return wikidata_id(uri)
    literal, _, datatype = uri.partition('^^')
    if datatype == _XSD + 'dateTime':
        return f'time:{literal.lstrip("+")[:10]}'
    if datatype == _XSD + 'decimal':
        from decimal import Decimal, InvalidOperation
        try:
            return f'quantity:{format(Decimal(literal.lstrip("+")).normalize(), "f")}'
        except InvalidOperation:
            pass
    return f'string:{literal.removeprefix("literal:")}'


def load_dataset(path: Path) -> list[dict[str, Any]]:
    LOG.info('Loading %s ...', path)
    start = time.perf_counter()
    data = path.read_bytes()
    try:
        import orjson
        docs = orjson.loads(data)
    except ImportError:
        docs = json.loads(data)
    if not isinstance(docs, list):
        raise SystemExit(f'{path}: expected a JSON list of T-REx documents')
    LOG.info('Loaded %d documents in %.1fs', len(docs), time.perf_counter() - start)
    return docs


def doc_key(doc: dict[str, Any]) -> str:
    return wikidata_id(doc['docid']) or doc['docid']


def truncate(doc: dict[str, Any], max_sentences: int | None) -> tuple[str, int]:
    """The document text, cut after `max_sentences` T-REx sentences."""
    text = doc['text']
    bounds = doc.get('sentences_boundaries') or []
    if max_sentences and len(bounds) > max_sentences:
        return text[:bounds[max_sentences - 1][1]], max_sentences
    return text, len(bounds)


def gold_triples(
        doc: dict[str, Any],
        annotators: set[str] | None = None,
        include_literals: bool = False,
        max_sentences: int | None = None,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """The distinct gold (subject, property, object) ids of a document.

    Values are the first occurrence's surface forms, for inspection.
    T-REx repeats a triple once per aligner; `annotators` keeps only the
    ones found by the given aligners (e.g. ``SPOAligner``). Literal objects
    (dates, quantities) are left out unless `include_literals`; they are
    keyed by `gold_value_key` (``time:1815-12-10``).
    """
    gold: dict[tuple[str, str, str], dict[str, Any]] = {}
    for t in doc.get('triples', []):
        if annotators and t.get('annotator') not in annotators:
            continue
        if max_sentences and (t.get('sentence_id') or 0) >= max_sentences:
            continue
        obj_iri = t['object']['uri']
        if not include_literals and 'wikidata.org/entity/' not in obj_iri:
            continue
        key = (wikidata_id(t['subject']['uri']),
               wikidata_id(t['predicate']['uri']),
               gold_value_key(obj_iri))
        gold.setdefault(key, {
            'subject': key[0], 'property': key[1], 'value': key[2],
            'surface': [t['subject'].get('surfaceform'),
                        t['predicate'].get('surfaceform'),
                        t['object'].get('surfaceform')],
            'sentence': t.get('sentence_id'),
        })
    return gold


STAGES = ('correct', 'extraction', 'entity_search', 'entity_disambiguation',
          'inverted_direction', 'relation_not_in_gold', 'property_search',
          'property_disambiguation', 'literal')
LITERAL_KINDS = ('literal', 'unknown', 'none')
GOLD_CAUSES = ('found', 'not_extracted', 'pair_not_extracted', 'relation_not_extracted',
               'entity_search', 'entity_disambiguation', 'inverted_direction',
               'property_search', 'property_disambiguation')

_ARTICLE = re.compile(r"^(the|a|an)\s+|'s$|’s$")
_DASH = re.compile(r'\s*([-‐‑‒–—])\s*')
PRONOUNS = {'he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'its', 'their', 'this',
            'that', 'these', 'those', 'who', 'which', 'we', 'i', 'you'}
EXTRACTION_REASONS = ('pronoun', 'literal', 'unannotated_span')


def normalize(term: str | None) -> str:
    text = ' '.join(str(term or '').lower().replace('_', ' ').split()).strip(' .,;:"\'()')
    return _DASH.sub(lambda m: '-', _ARTICLE.sub('', text)).strip()


def extraction_reason(term: str) -> str:
    """Why a subject/object that is not an annotated entity was extracted."""
    text = normalize(term)
    if text in PRONOUNS:
        return 'pronoun'
    if re.search(r'\d', text) or re.fullmatch(
            r'(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|'
            r'twenty|hundred|thousand|million|billion)(\s+\w+)?', text):
        return 'literal'
    return 'unannotated_span'


def gold_aliases(doc: dict[str, Any], gold: dict) -> dict[str, set[str]]:
    """Normalized surface form -> ids of the annotated entities it names."""
    aliases: dict[str, set[str]] = {}

    def add(name: str | None, iri_or_id: str | None) -> None:
        qid = wikidata_id(iri_or_id)
        if name and qid and qid.startswith('Q'):
            aliases.setdefault(normalize(name), set()).add(qid)

    for e in doc.get('entities') or []:
        add(e.get('surfaceform'), e.get('uri'))
    for g in gold.values():
        add(g['surface'][0], g['subject'])
        add(g['surface'][2], g['value'])
    aliases.pop('', None)
    return aliases


def diagnose(extracted: list[dict[str, Any]], gold: dict, aliases: dict[str, set[str]],
             candidates_of: Any) -> dict[tuple[str, str, str], str]:
    """Labels every extracted triple with the stage where it went wrong.

    Sets ``stage`` (one of `STAGES`) and ``diagnosis`` (per part) on each
    extracted triple, by matching its subject/object to the annotated
    entities (`aliases`) and checking the search candidates and the chosen
    links. A subject/object is right when it is linked to the entity its
    span is annotated with, or, for a span that is not annotated, to any
    annotated entity of the document. A triple whose statement is in the
    gold is always `correct`.

    Returns the cause (one of `GOLD_CAUSES`) of every gold triple.
    `candidates_of(term, entity_type)` gives the ids found by the search.
    """
    props_of: dict[tuple[str, str], set[str]] = {}
    for s, p, o in gold:
        props_of.setdefault((s, o), set()).add(p)
    order = {stage: i for i, stage in enumerate(STAGES)}
    annotated = {qid for ids in aliases.values() for qid in ids}

    def entity(term: str, link: dict[str, Any]) -> tuple[str, set[str]]:
        expected = aliases.get(normalize(term), set())
        if link['id'] in expected:
            return 'correct', expected
        if not expected:
            if link['id'] in annotated:
                return 'correct', {link['id']}
            return 'extraction', expected
        if expected & set(candidates_of(term, 'item') or []):
            return 'entity_disambiguation', expected
        return 'entity_search', expected

    for t in extracted:
        literal_value = t.get('object_kind') in LITERAL_KINDS
        s_stage, s_ids = entity(t['subject'], t['linked']['subject'])
        o_stage, o_ids = (('correct', set()) if literal_value
                          else entity(t['object'], t['linked']['object']))
        object_id = t['linked']['object'].get('key') or t['linked']['object']['id']
        p_stage = 'not_evaluated'
        stage = max((s_stage, o_stage), key=lambda x: (x != 'correct', -order[x]))
        if tuple(t['statement'] or ()) in gold:
            stage = s_stage = o_stage = p_stage = 'correct'
            s_ids = {t['linked']['subject']['id']} | s_ids
            o_ids = o_ids if literal_value else {t['linked']['object']['id']} | o_ids
        elif stage == 'correct':
            pair = (t['linked']['subject']['id'], object_id)
            props = props_of.get(pair)
            if props is None:
                stage = ('literal' if literal_value else 'inverted_direction'
                         if (pair[1], pair[0]) in props_of else 'relation_not_in_gold')
            elif t['linked']['predicate']['id'] in props:
                stage = p_stage = 'correct'
            elif props & set(candidates_of(t['predicate'], 'property') or []):
                stage = p_stage = 'property_disambiguation'
            else:
                stage = p_stage = 'property_search'
        t['stage'] = stage
        t['diagnosis'] = {'subject': s_stage, 'predicate': p_stage, 'object': o_stage,
                          'gold_subject': sorted(s_ids), 'gold_object': sorted(o_ids)}
        if stage == 'extraction':
            reasons = {extraction_reason(t[k]) for k, x in (('subject', s_stage), ('object', o_stage))
                       if x == 'extraction'}
            t['diagnosis']['extraction'] = next(r for r in EXTRACTION_REASONS if r in reasons)

    causes = {}
    for key in gold:
        s, _, o = key
        if any(tuple(t['statement'] or ()) == key for t in extracted):
            causes[key] = 'found'
            continue
        same_pair = [t for t in extracted if s in t['diagnosis']['gold_subject']
                     and o in t['diagnosis']['gold_object']]
        if same_pair:
            cause = max((t['stage'] for t in same_pair), key=lambda x: order[x])
            causes[key] = {'correct': 'relation_not_extracted',
                           'relation_not_in_gold': 'entity_disambiguation'}.get(cause, cause)
        elif any(o in t['diagnosis']['gold_subject'] and s in t['diagnosis']['gold_object']
                 for t in extracted):
            causes[key] = 'inverted_direction'
        else:
            mentioned = {q for t in extracted
                         for q in t['diagnosis']['gold_subject'] + t['diagnosis']['gold_object']}
            causes[key] = 'pair_not_extracted' if {s, o} <= mentioned else 'not_extracted'
    return causes


def select_documents(docs: list[dict[str, Any]], args: argparse.Namespace) -> list[dict]:
    selected = docs
    if args.docids:
        wanted = parse_docids(args.docids)
        by_key = {doc_key(d): d for d in docs}
        missing = [k for k in wanted if k not in by_key]
        if missing:
            LOG.warning('%d docids not in the dataset: %s', len(missing),
                        ', '.join(missing[:10]))
        selected = [by_key[k] for k in wanted if k in by_key]
    if args.min_chars:
        selected = [d for d in selected if len(d['text']) >= args.min_chars]
    if args.max_chars:
        selected = [d for d in selected if len(d['text']) <= args.max_chars]
    if args.min_gold:
        selected = [d for d in selected if len(gold_triples(
            d, args.gold_annotators, args.include_literals,
            args.max_sentences)) >= args.min_gold]
    if args.sample:
        rng = random.Random(args.seed)
        selected = rng.sample(selected, min(args.sample, len(selected)))
    selected = selected[args.offset:]
    if args.limit is not None:
        selected = selected[:args.limit]
    return selected


def parse_docids(spec: str) -> list[str]:
    """`Q1,Q2` or `@file` (one id or IRI per line)."""
    if spec.startswith('@'):
        items = Path(spec[1:]).read_text().split()
    else:
        items = spec.split(',')
    return [wikidata_id(i.strip()) for i in items if i.strip()]


def prf(predicted: set, gold: set) -> dict[str, Any]:
    tp = len(predicted & gold)
    p = tp / len(predicted) if predicted else 0.0
    r = tp / len(gold) if gold else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {'tp': tp, 'predicted': len(predicted), 'gold': len(gold),
            'precision': p, 'recall': r, 'f1': f1}


def evaluate(predicted: set[tuple[str, str, str]],
             gold: set[tuple[str, str, str]]) -> dict[str, dict[str, Any]]:
    """Strict triples, plus two relaxed views for error analysis.

    - ``strict``: (subject, property, object) must all match.
    - ``entity_pair``: (subject, object) match, whatever the property.
    - ``entities``: entities of the predicted statements vs gold entities
      (literal values are not entities).
    """
    return {
        'strict': prf(predicted, gold),
        'entity_pair': prf({(s, o) for s, _, o in predicted},
                           {(s, o) for s, _, o in gold}),
        'entities': prf({e for s, _, o in predicted for e in (s, o) if is_entity_id(e)},
                        {e for s, _, o in gold for e in (s, o) if is_entity_id(e)}),
    }


def micro(rows: Iterable[dict[str, Any]], level: str) -> dict[str, Any]:
    tp = pred = gold = 0
    for row in rows:
        m = row['metrics'][level]
        tp, pred, gold = tp + m['tp'], pred + m['predicted'], gold + m['gold']
    p = tp / pred if pred else 0.0
    r = tp / gold if gold else 0.0
    return {'tp': tp, 'predicted': pred, 'gold': gold, 'precision': p,
            'recall': r, 'f1': 2 * p * r / (p + r) if p + r else 0.0}


@dataclass
class Trace:
    """Calls made by `aconstruct` for one document (times from its start)."""
    t0: float = field(default_factory=time.perf_counter)
    extractions: list[dict[str, Any]] = field(default_factory=list)
    searches: list[dict[str, Any]] = field(default_factory=list)
    disambiguations: list[dict[str, Any]] = field(default_factory=list)
    coreferences: list[dict[str, Any]] = field(default_factory=list)
    objects: list[dict[str, Any]] = field(default_factory=list)
    resolved: str | None = None
    first_statement_s: float | None = None

    def now(self) -> float:
        return time.perf_counter() - self.t0


def _error(e: BaseException) -> dict[str, str]:
    return {'type': type(e).__name__, 'message': str(e)[:2000]}


def _candidate(c: Any, descriptions: bool = True) -> dict[str, Any]:
    out = {'id': wikidata_id(c.iri), 'label': c.label}
    if descriptions:
        out['description'] = c.description
    return out


class Probe:
    """Wraps the extractor, the search and the disambiguator of a `KBEL`.

    Each call is recorded — inputs, outputs, start/end, errors — into the
    `Trace` active when the call *started*, so calls still running after a
    timeout never leak into the next document.
    """

    def __init__(self, kbel: Any):
        self.trace: Trace | None = None
        extractor, disambiguator = kbel.extractor, kbel.disambiguator

        original_aextract_batch = extractor.aextract_batch
        original_lookup = kbel.candidates_lookup
        original_adisambiguate = disambiguator.adisambiguate
        coreference = kbel.coreference
        original_aresolve = coreference.aresolve if coreference is not None else None

        async def aextract_batch(texts: list[str], *args: Any, **kwargs: Any):
            trace = self.trace
            rec: dict[str, Any] = {'sentences': list(texts), 'start': trace.now()}
            try:
                batch = await original_aextract_batch(texts, *args, **kwargs)
                rec['triples'] = [[[t.subject, t.predicate, t.object] for t in triples]
                                  for triples in batch]
                return batch
            except BaseException as e:
                rec['error'] = _error(e)
                raise
            finally:
                rec['end'] = trace.now()
                trace.extractions.append(rec)

        def candidates_lookup(mention: Any, limit: int | None = None):
            trace = self.trace
            rec: dict[str, Any] = {'term': mention.term,
                                   'entity_type': mention.entity_type.value,
                                   'start': trace.now()}
            try:
                candidates = original_lookup(mention, limit)
                rec['candidates'] = candidates
                return candidates
            except BaseException as e:
                rec['error'] = _error(e)
                raise
            finally:
                rec['end'] = trace.now()
                trace.searches.append(rec)

        async def adisambiguate(mention: Any, candidates: list, *args: Any, **kwargs: Any):
            trace = self.trace
            rec: dict[str, Any] = {'term': mention.term,
                                   'entity_type': mention.entity_type.value,
                                   'sentence': mention.text,
                                   'context': mention.context,
                                   'n_candidates': len(candidates),
                                   'start': trace.now()}
            try:
                results = await original_adisambiguate(mention, candidates, *args, **kwargs)
                rec['results'] = results
                return results
            except BaseException as e:
                rec['error'] = _error(e)
                raise
            finally:
                rec['end'] = trace.now()
                trace.disambiguations.append(rec)

        original_to_statement = kbel._to_statement

        def to_statement(linked: Any):
            trace = self.trace
            statement = original_to_statement(linked)
            obj = linked.object[0] if linked.object else None
            info: dict[str, Any] = {
                'triple': [linked.triple.subject, linked.triple.predicate, linked.triple.object]}
            if linked.snak is not None:
                info['kind'] = 'unknown' if linked.snak == 'some' else 'none'
            elif obj is not None and obj.is_literal:
                info['kind'] = 'literal'
                info['key'] = value_key(obj.entity)
            elif obj is not None:
                info['kind'] = 'entity'
                info['id'] = wikidata_id(obj.candidate.iri)
            trace.objects.append(info)
            return statement

        async def aresolve(text: str):
            trace = self.trace
            rec: dict[str, Any] = {'chars': len(text), 'start': trace.now()}
            try:
                resolved = await original_aresolve(text)
                rec['changed'] = resolved != text
                trace.resolved = resolved
                return resolved
            except BaseException as e:
                rec['error'] = _error(e)
                raise
            finally:
                rec['end'] = trace.now()
                trace.coreferences.append(rec)

        extractor.aextract_batch = aextract_batch
        kbel.candidates_lookup = candidates_lookup
        disambiguator.adisambiguate = adisambiguate
        kbel._to_statement = to_statement
        if coreference is not None:
            coreference.aresolve = aresolve


def _active_time(intervals: list[tuple[float, float]]) -> float:
    """Length of the union of the intervals (time with >= 1 call running)."""
    total, end = 0.0, float('-inf')
    for s, e in sorted(intervals):
        if s > end:
            total += e - s
            end = e
        elif e > end:
            total += e - end
            end = e
    return total


def stage_timing(records: list[dict[str, Any]]) -> dict[str, Any]:
    intervals = [(r['start'], r['end']) for r in records]
    durations = [e - s for s, e in intervals]
    return {
        'calls': len(records),
        'errors': sum('error' in r for r in records),
        'busy_s': round(sum(durations), 4),
        'active_s': round(_active_time(intervals), 4),
        'max_call_s': round(max(durations), 4) if durations else 0.0,
        'first_start_s': round(min(s for s, _ in intervals), 4) if intervals else None,
        'last_end_s': round(max(e for _, e in intervals), 4) if intervals else None,
    }


def _statement_ids(statement: Any) -> tuple[str, str, str]:
    """``(subject id, property id, value key)`` of a statement (see `value_key`)."""
    from kif_lib import SomeValueSnak, ValueSnak

    snak = statement.snak
    if isinstance(snak, ValueSnak):
        value = value_key(snak.value)
    else:
        value = 'unknown' if isinstance(snak, SomeValueSnak) else 'none'
    return (wikidata_id(statement.subject.iri.content),
            wikidata_id(snak.property.iri.content), value)


async def construct_document(kbel: Any, probe: Probe, text: str,
                             args: argparse.Namespace) -> tuple[list, Trace, float]:
    """Runs `aconstruct` on `text`; returns (statements, trace, total_s)."""
    trace = probe.trace = Trace()
    statements: list[Any] = []

    async def consume() -> None:
        async for statement in kbel.aconstruct(
                text, language=args.language, search_limit=args.search_limit,
                max_concurrency=args.max_concurrency):
            if trace.first_statement_s is None:
                trace.first_statement_s = trace.now()
            statements.append(statement)

    try:
        await asyncio.wait_for(consume(), timeout=args.timeout)
    except BaseException as e:
        e.partial = (statements, trace, trace.now())   # type: ignore[attr-defined]
        raise
    return statements, trace, trace.now()


def build_record(doc: dict[str, Any], text: str, n_sentences: int,
                 statements: list[Any], trace: Trace, total_s: float,
                 split_s: float, units: list[tuple[str, str | None]],
                 args: argparse.Namespace) -> dict[str, Any]:
    """The JSONL line of a document (without run metadata)."""
    from kbel.core.kbel import KBEL
    from kbel.extractors import TripleExtractor
    from kbel.core.triple import Triple

    if trace.resolved is not None:
        units = KBEL.sentences_with_context(trace.resolved)
    gold = gold_triples(doc, args.gold_annotators, args.include_literals,
                        args.max_sentences)
    predicted = [_statement_ids(s) for s in statements]
    gold_set = set(gold)
    counted = [ids for ids in predicted if args.include_literals or is_entity_id(ids[2])]
    predicted_set = set(counted)
    metrics = evaluate(predicted_set, gold_set)

    extraction_of: dict[str, list] = {}
    for rec in trace.extractions:
        for sentence, triples in zip(rec['sentences'], rec.get('triples') or []):
            extraction_of.setdefault(sentence, triples)
    index_of: dict[str, int] = {}
    for i, (sentence, _) in enumerate(units):
        index_of.setdefault(sentence, i)
    search_of = {}
    for rec in trace.searches:
        search_of.setdefault((rec['term'], rec['entity_type']), rec)
    link_of = {}
    for rec in trace.disambiguations:
        link_of.setdefault((rec['term'], rec['entity_type'], rec['sentence'],
                            rec['context']), rec)

    def part(mention: Any) -> dict[str, Any]:
        key = (mention.term, mention.entity_type.value)
        search = search_of.get(key)
        if search is None:
            return {'id': None, 'reason': 'not_searched'}
        if 'error' in search:
            return {'id': None, 'reason': 'search_error'}
        if not search['candidates']:
            return {'id': None, 'reason': 'no_candidates'}
        link = link_of.get((*key, mention.text, mention.context))
        if link is None:
            return {'id': None, 'reason': 'not_disambiguated'}
        if 'error' in link:
            return {'id': None, 'reason': 'disambiguation_error'}
        if not link['results']:
            return {'id': None, 'reason': 'no_result'}
        best = link['results'][0]
        return {'id': wikidata_id(best.candidate.iri), 'label': best.candidate.label,
                'score': best.score, 'rank_in_search':
                    next((i for i, c in enumerate(search['candidates'])
                          if c.iri == best.candidate.iri), None)}

    captured_objects = dict(enumerate(trace.objects))
    extracted = []
    for index, (sentence, paragraph) in enumerate(units):
        if sentence not in extraction_of:
            continue
        triples = [Triple(*t) for t in extraction_of[sentence]]
        for t, mentions in zip(triples, TripleExtractor.build_mentions(
                triples, sentence, context=paragraph, language=args.language)):
            linked = dict(zip(('subject', 'predicate', 'object'), map(part, mentions)))
            kind = 'entity'
            captured = captured_objects.get(len(extracted))
            if captured is not None and captured['triple'] == [t.subject, t.predicate, t.object]:
                kind = captured.get('kind', 'unlinked')
                if kind in ('literal', 'unknown', 'none'):
                    linked['object'] = {'id': None, kind: captured.get('key', kind)}
                    linked['object']['key'] = captured.get('key', kind)
                elif kind == 'entity' and captured.get('id') != linked['object'].get('id'):
                    linked['object'] = {'id': captured['id'], 'label': None}
            ids = (linked['subject']['id'], linked['predicate']['id'],
                   linked['object'].get('key') or linked['object']['id'])
            complete = all(ids)
            evaluated = complete and (args.include_literals or is_entity_id(ids[2]))
            extracted.append({
                'sentence': index,
                'subject': t.subject, 'predicate': t.predicate, 'object': t.object,
                'linked': linked,
                'object_kind': kind,
                'statement': list(ids) if complete else None,
                'correct': (ids in gold_set) if evaluated else None,
            })

    def candidates_of(term: str, entity_type: str) -> list[str]:
        rec = search_of.get((term, entity_type))
        return [wikidata_id(c.iri) for c in (rec or {}).get('candidates') or []]

    causes = diagnose(extracted, gold, gold_aliases(doc, gold), candidates_of)

    labels = {}
    for rec in trace.searches:
        for c in rec.get('candidates') or []:
            labels.setdefault(wikidata_id(c.iri), c.label)

    timing = {
        'split_s': round(split_s, 4),
        'extraction': {**stage_timing(trace.extractions),
                       'sentences': sum(len(r['sentences']) for r in trace.extractions)},
        'coreference': stage_timing(trace.coreferences),
        'search': stage_timing(trace.searches),
        'disambiguation': stage_timing(trace.disambiguations),
        'first_statement_s': (round(trace.first_statement_s, 4)
                              if trace.first_statement_s is not None else None),
        'total_s': round(total_s, 4),
    }

    record: dict[str, Any] = {
        'docid': doc_key(doc),
        'uri': doc.get('uri') or doc['docid'],
        'title': doc.get('title'),
        'n_chars': len(text),
        'n_sentences_trex': n_sentences,
        'n_sentences_kbel': len(units),
        'precision': metrics['strict']['precision'],
        'recall': metrics['strict']['recall'],
        'f1': metrics['strict']['f1'],
        'metrics': metrics,
        'timing': timing,
        'extracted_triples': extracted,
        'statements': [{'subject': s, 'property': p, 'value': o,
                        'labels': [labels.get(s), labels.get(p), labels.get(o)],
                        'correct': ((s, p, o) in gold_set
                                    if args.include_literals or is_entity_id(o) else None)}
                       for s, p, o in predicted],
        'gold': [{**g, 'found': k in set(predicted), 'cause': causes[k]} for k, g in gold.items()],
    }
    if trace.resolved is not None:
        record['coreference'] = {'resolved_text': trace.resolved,
                                 'changed': trace.resolved != text}
    if args.trace != 'none':
        full = args.trace == 'full'
        record['trace'] = {
            'coreferences': [_round(r) for r in trace.coreferences],
            'extractions': [_round(r) for r in trace.extractions],
            'searches': [_round({**r, 'candidates': [
                _candidate(c, full) for c in r.get('candidates') or []]})
                for r in trace.searches],
            'disambiguations': [_round({
                **{k: v for k, v in r.items() if k not in ('results', 'context')},
                'sentence': index_of.get(r['sentence'], r['sentence']),
                'results': [{**_candidate(x.candidate, False), 'score': x.score}
                            for x in r.get('results') or []],
            }) for r in trace.disambiguations],
        }
    return record


def _round(rec: dict[str, Any]) -> dict[str, Any]:
    return {k: round(v, 4) if k in ('start', 'end') else v for k, v in rec.items()}


async def evaluate_document(kbel: Any, probe: Probe, doc: dict[str, Any],
                            args: argparse.Namespace) -> dict[str, Any]:
    """Evaluates one document, retrying failures; never raises (except ^C)."""
    from kbel.core.kbel import KBEL

    text, n_sentences = truncate(doc, args.max_sentences)
    start = time.perf_counter()
    units = KBEL.sentences_with_context(text)
    split_s = time.perf_counter() - start

    attempts: list[dict[str, Any]] = []
    for attempt in range(1, args.retries + 2):
        try:
            statements, trace, total_s = await construct_document(kbel, probe, text, args)
            record = build_record(doc, text, n_sentences, statements, trace,
                                  total_s, split_s, units, args)
            record['status'] = 'ok'
            break
        except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
            raise
        except Exception as e:
            status = 'timeout' if isinstance(e, (TimeoutError, asyncio.TimeoutError)) else 'error'
            attempts.append({**_error(e), 'status': status,
                             'traceback': traceback.format_exc(limit=20)})
            LOG.warning('%s: attempt %d failed (%s: %s)', doc_key(doc), attempt,
                        type(e).__name__, str(e)[:200])
            if attempt <= args.retries:
                await asyncio.sleep(min(args.retry_backoff * 2 ** (attempt - 1), 60))
                continue
            statements, trace, total_s = getattr(e, 'partial', ([], Trace(), 0.0))
            record = build_record(doc, text, n_sentences, statements, trace,
                                  total_s, split_s, units, args)
            record['status'] = status
            record['error'] = attempts[-1]
    record['attempts'] = len(attempts) + (record['status'] == 'ok')
    if attempts:
        record['failed_attempts'] = attempts
    return record


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Valid records of `path`; corrupt lines (e.g. a crash mid-write) are skipped."""
    if not path.exists():
        return
    with path.open(encoding='utf-8') as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                LOG.warning('%s:%d: skipping corrupt line', path, n)


def latest_records(path: Path) -> dict[str, dict[str, Any]]:
    """Last record of every docid (later lines supersede earlier ones)."""
    return {r['docid']: r for r in read_jsonl(path) if 'docid' in r}


class Output:
    """Append-only JSONL writer; each line is flushed and fsynced."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        needs_newline = (path.exists() and path.stat().st_size > 0
                         and not path.read_bytes().endswith(b'\n'))
        self._f = path.open('a', encoding='utf-8')
        if needs_newline:
            self._f.write('\n')

    def write(self, record: dict[str, Any]) -> None:
        self._f.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
        self._f.flush()
        os.fsync(self._f.fileno())

    def close(self) -> None:
        self._f.close()


@contextlib.contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    """Refuses to run twice on the same output at the same time."""
    lock_path = path.with_name(path.name + '.lock')
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    f = lock_path.open('w')
    try:
        try:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except ImportError:
            pass
        except BlockingIOError:
            raise SystemExit(f'{path} is being written by another run '
                             f'(lock: {lock_path}).')
        f.write(str(os.getpid()))
        f.flush()
        yield
    finally:
        f.close()
        with contextlib.suppress(OSError):
            lock_path.unlink()


def backup(path: Path) -> Path:
    stamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    target = path.with_name(f'{path.name}.{stamp}.bak')
    path.rename(target)
    return target


def parse_options(pairs: list[str] | None) -> dict[str, Any]:
    """`['num_beams=6', 'x={"a": 1}']` -> {'num_beams': 6, 'x': {'a': 1}}."""
    options: dict[str, Any] = {}
    for pair in pairs or []:
        key, sep, value = pair.partition('=')
        if not sep or not key:
            raise SystemExit(f'Invalid option {pair!r}: expected KEY=VALUE')
        try:
            options[key.strip()] = json.loads(value)
        except json.JSONDecodeError:
            options[key.strip()] = value
    return options


def llm_api_key(args: argparse.Namespace) -> tuple[str, str]:
    names = [args.llm_api_key_env] if args.llm_api_key_env else list(DEFAULT_LLM_KEY_ENVS)
    for name in names:
        if os.environ.get(name):
            return name, os.environ[name]
    raise SystemExit(f'The LLM plugins need an API key: set {" or ".join(names)} '
                     f'(environment or {args.env_file}).')


def plugin_options(name: str, options: dict[str, Any],
                   args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """Plugin kwargs and their loggable version (API key replaced by its env name).

    LLM plugins default to OpenRouter: the `llm` ones through its
    OpenAI-compatible API (LangChain), KGGen through LiteLLM's
    ``openrouter/`` models.
    """
    if name in LITELLM_PLUGINS and not {'model', 'kg'} & set(options):
        key_env, key = llm_api_key(args)
        if args.llm_endpoint.rstrip('/') == OPENROUTER_ENDPOINT:
            target = {'model': f'openrouter/{args.llm_model}'}
        else:
            target = {'model': f'openai/{args.llm_model}', 'api_base': args.llm_endpoint}
        options = {**target, 'temperature': args.llm_temperature, **options}
        return {**options, 'api_key': key}, {**options, 'api_key': f'<env:{key_env}>'}
    if name not in LANGCHAIN_LLM_PLUGINS or 'model' in options:
        return dict(options), dict(options)
    key_env, key = llm_api_key(args)
    params = {'temperature': args.llm_temperature, **options.pop('model_params', {})}
    options = {'model_name': args.llm_model, 'model_provider': args.llm_provider,
               'model_endpoint': args.llm_endpoint, 'model_params': params, **options}
    return ({**options, 'model_apikey': key},
            {**options, 'model_apikey': f'<env:{key_env}>'})


def fingerprint(args: argparse.Namespace, ext_log: dict, dis_log: dict,
                kb_opts: dict, coref_opts: dict | None = None) -> dict[str, Any]:
    """Everything that changes the results of a document."""
    dataset = Path(args.dataset)
    coreference = {'coreference': args.coreference, 'coreference_options': coref_opts or {}} \
        if args.coreference else {}
    return {
        **coreference,
        'schema': SCHEMA_VERSION,
        'dataset': dataset.name,
        'dataset_bytes': dataset.stat().st_size if dataset.exists() else None,
        'knowledge_base': args.kb, 'kb_options': kb_opts,
        'extractor': args.extractor, 'extractor_options': ext_log,
        'disambiguator': args.disambiguator, 'disambiguator_options': dis_log,
        'search_limit': args.search_limit, 'language': args.language,
        'max_sentences': args.max_sentences,
        'gold_annotators': sorted(args.gold_annotators or []),
        'include_literals': args.include_literals,
    }


def config_hash(fp: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(fp, sort_keys=True, default=str)
                          .encode()).hexdigest()[:12]


def git_state() -> dict[str, Any]:
    def git(*cmd: str) -> str:
        return subprocess.run(['git', *cmd], cwd=ROOT, capture_output=True,
                              text=True, timeout=10).stdout.strip()
    try:
        return {'commit': git('rev-parse', 'HEAD') or None,
                'dirty': bool(git('status', '--porcelain', '--untracked-files=no'))}
    except Exception:
        return {'commit': None, 'dirty': None}


def check_config(path: Path, fp: dict[str, Any], force: bool) -> None:
    """Refuses to append results of a different configuration."""
    if not path.exists():
        return
    previous = json.loads(path.read_text())['fingerprint']
    diff = {k: (previous.get(k), fp.get(k)) for k in set(previous) | set(fp)
            if previous.get(k) != fp.get(k)}
    if diff and not force:
        lines = '\n'.join(f'  {k}: {old!r} -> {new!r}' for k, (old, new) in sorted(diff.items()))
        raise SystemExit(
            f'{path.name.removesuffix(".config.json")} was produced with another configuration:\n{lines}\n'
            'Use another --output, --overwrite (the old file is backed up) or '
            '--force (mixes configurations).')


def default_output(args: argparse.Namespace) -> Path:
    parts = [args.dataset_name, args.kb, args.extractor, args.disambiguator]
    if args.coreference:
        parts.insert(2, args.coreference)
    if (LANGCHAIN_LLM_PLUGINS | LITELLM_PLUGINS) & {args.extractor, args.disambiguator}:
        parts.append(args.llm_model)
    if args.max_sentences:
        parts.append(f's{args.max_sentences}')
    name = '_'.join(re.sub(r'[^A-Za-z0-9.-]+', '-', p) for p in parts)
    return Path(args.output_dir) / f'{name}.jsonl'


def build_kbel(args: argparse.Namespace, ext_opts: dict, dis_opts: dict,
               kb_opts: dict, coref_opts: dict | None = None) -> Any:
    from kif_lib.search import Search
    from kbel.core.kbel import KBEL
    from kbel.coreference import CoreferenceResolver
    from kbel.disambiguators import Disambiguator
    from kbel.extractors import TripleExtractor

    for module in [EXTRACTOR_MODULES.get(args.extractor),
                   DISAMBIGUATOR_MODULES.get(args.disambiguator),
                   COREFERENCE_MODULES.get(args.coreference), *args.plugin]:
        if module:
            importlib.import_module(module)
    for kind, registry, name in [('extractor', TripleExtractor.registry, args.extractor),
                                 ('disambiguator', Disambiguator.registry, args.disambiguator)]:
        if name not in registry:
            raise SystemExit(f'Unknown {kind} {name!r}; available: {", ".join(sorted(registry))} '
                             '(custom plugins: --plugin MODULE)')
    if args.coreference and args.coreference not in CoreferenceResolver.registry:
        raise SystemExit(f'Unknown coreference resolver {args.coreference!r}; available: '
                         f'{", ".join(sorted(CoreferenceResolver.registry))}')
    if args.kb not in Search.registry:
        raise SystemExit(f'Unknown knowledge base {args.kb!r}; available: '
                         f'{", ".join(sorted(Search.registry))}')

    start = time.perf_counter()
    kbel = KBEL(Search(args.kb, limit=args.search_limit, **kb_opts),
                search_limit=args.search_limit)
    kbel.extractor = TripleExtractor(args.extractor, **ext_opts)
    kbel.disambiguator = Disambiguator(args.disambiguator, **dis_opts)
    if args.coreference:
        kbel.coreference = CoreferenceResolver(args.coreference, **(coref_opts or {}))
    LOG.info('Pipeline ready in %.1fs', time.perf_counter() - start)
    return kbel


def _short(record: dict[str, Any]) -> str:
    m, t = record['metrics']['strict'], record['timing']
    return (f'{record["status"]:<7} P={m["precision"]:.2f} R={m["recall"]:.2f} '
            f'F1={m["f1"]:.2f} ({m["tp"]}/{m["predicted"]} pred, {m["gold"]} gold) '
            f'{t["total_s"]:.1f}s')


async def run_documents(kbel: Any, docs: list[dict], output: Output,
                        args: argparse.Namespace, meta: dict[str, Any]) -> dict[str, int]:
    probe = Probe(kbel)
    counts = {'ok': 0, 'error': 0, 'timeout': 0}
    consecutive_failures = 0
    rows: list[dict[str, Any]] = []

    progress = None
    if not args.no_progress:
        with contextlib.suppress(ImportError):
            from tqdm import tqdm
            progress = tqdm(total=len(docs), unit='doc', dynamic_ncols=True)

    try:
        for i, doc in enumerate(docs, 1):
            record = await evaluate_document(kbel, probe, doc, args)
            record.update(meta, finished_at=dt.datetime.now().isoformat(timespec='seconds'))
            output.write(record)

            counts[record['status']] += 1
            if record['status'] == 'ok':
                rows.append(record)
                consecutive_failures = 0
            else:
                consecutive_failures += 1
            line = f'[{i}/{len(docs)}] {record["docid"]} {_short(record)}'
            if progress is not None:
                m = micro(rows, 'strict') if rows else None
                progress.set_postfix_str(
                    f'F1={m["f1"]:.3f} err={counts["error"] + counts["timeout"]}' if m else '')
                progress.update()
                LOG.debug(line)
            else:
                LOG.info(line)

            if args.max_consecutive_errors and consecutive_failures >= args.max_consecutive_errors:
                LOG.error('%d documents failed in a row; stopping (see "error" in %s). '
                          'Fix the problem and run the same command to resume.',
                          consecutive_failures, output.path)
                counts['aborted'] = 1
                break
    finally:
        if progress is not None:
            progress.close()
    return counts


def cmd_run(args: argparse.Namespace) -> int:
    if args.env_file and Path(args.env_file).exists():
        with contextlib.suppress(ImportError):
            from dotenv import load_dotenv
            load_dotenv(args.env_file, override=False)

    args.gold_annotators = set(args.gold_annotators) if args.gold_annotators else None
    ext_opts, ext_log = plugin_options(args.extractor, parse_options(args.extractor_opt), args)
    dis_opts, dis_log = plugin_options(args.disambiguator, parse_options(args.disambiguator_opt), args)
    kb_opts = parse_options(args.kb_opt)
    coref_opts = parse_options(args.coreference_opt)

    output_path = Path(args.output) if args.output else default_output(args)
    config_path = output_path.with_name(output_path.name + '.config.json')
    fp = fingerprint(args, ext_log, dis_log, kb_opts, coref_opts)
    chash = config_hash(fp)

    with exclusive_lock(output_path):
        if args.overwrite:
            for p in (output_path, config_path):
                if p.exists():
                    LOG.warning('Backed up %s to %s', p, backup(p))
        check_config(config_path, fp, args.force)

        done = latest_records(output_path)
        skip = {k for k, r in done.items()
                if r.get('status') == 'ok' or not args.retry_failed}

        docs = select_documents(args.loader(Path(args.dataset)), args)
        todo = [d for d in docs if doc_key(d) not in skip]
        LOG.info('%d documents selected, %d already in %s, %d to run',
                 len(docs), len(docs) - len(todo), output_path, len(todo))
        if args.dry_run:
            print(json.dumps({'output': str(output_path), 'config_hash': chash,
                              'fingerprint': fp, 'to_run': [doc_key(d) for d in todo]},
                             indent=2, default=str))
            return 0
        if not todo:
            LOG.info('Nothing to do.')
            return 0

        kbel = build_kbel(args, ext_opts, dis_opts, kb_opts, coref_opts)

        config = {'fingerprint': fp, 'config_hash': chash,
                  'created_at': dt.datetime.now().isoformat(timespec='seconds')}
        if config_path.exists():
            config = json.loads(config_path.read_text())
        config.setdefault('invocations', []).append({
            'started_at': dt.datetime.now().isoformat(timespec='seconds'),
            'argv': sys.argv[1:], 'git': git_state(), 'python': platform.python_version(),
            'host': platform.node(), 'max_concurrency': args.max_concurrency,
            'timeout': args.timeout, 'documents': len(todo)})
        tmp = config_path.with_name(config_path.name + '.tmp')
        tmp.write_text(json.dumps(config, indent=2, default=str))
        tmp.replace(config_path)

        log_file = logging.FileHandler(output_path.with_name(output_path.name + '.log'))
        log_file.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
        logging.getLogger().addHandler(log_file)

        meta = {'config_hash': chash, 'max_concurrency': args.max_concurrency}
        output = Output(output_path)
        try:
            counts = asyncio.run(run_documents(kbel, todo, output, args, meta))
        except KeyboardInterrupt:
            LOG.warning('Interrupted. Finished documents are saved in %s; '
                        'run the same command to resume.', output_path)
            return 130
        finally:
            output.close()
            logging.getLogger().removeHandler(log_file)
            log_file.close()

    LOG.info('Done: %s. Results in %s', ', '.join(f'{v} {k}' for k, v in counts.items()),
             output_path)
    if not args.no_summary:
        print_summary([output_path], common=False, fmt='table')
    return 1 if counts.get('aborted') else 0


STAGE_NAMES = ('coreference', 'extraction', 'search', 'disambiguation')


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def summarize(path: Path, docids: set[str] | None = None) -> dict[str, Any]:
    records = latest_records(path)
    if docids is not None:
        records = {k: v for k, v in records.items() if k in docids}
    ok = [r for r in records.values() if r.get('status') == 'ok']
    with_gold = [r for r in ok if r['metrics']['strict']['gold']]
    config = path.with_name(path.name + '.config.json')
    fp = json.loads(config.read_text())['fingerprint'] if config.exists() else {}

    def stage(name: str, key: str) -> float | None:
        return _mean([r['timing'].get(name, {}).get(key, 0.0) for r in ok])

    summary: dict[str, Any] = {
        'file': str(path),
        'knowledge_base': fp.get('knowledge_base'), 'extractor': fp.get('extractor'),
        'coreference': fp.get('coreference'),
        'disambiguator': fp.get('disambiguator'),
        'documents': len(records), 'ok': len(ok),
        'error': sum(r.get('status') == 'error' for r in records.values()),
        'timeout': sum(r.get('status') == 'timeout' for r in records.values()),
        'extracted_triples': sum(len(r['extracted_triples']) for r in ok),
        'linked_triples': sum(r['statement'] is not None
                              for x in ok for r in x['extracted_triples']),
    }
    for level in ('strict', 'entity_pair', 'entities'):
        m = micro(ok, level)
        summary[level] = {
            'micro': m,
            'macro': {k: _mean([r['metrics'][level][k] for r in with_gold])
                      for k in ('precision', 'recall', 'f1')},
        }
    summary['timing'] = {
        'total_s_mean': _mean([r['timing']['total_s'] for r in ok]),
        'total_s_median': statistics.median([r['timing']['total_s'] for r in ok]) if ok else None,
        'total_s_sum': sum(r['timing']['total_s'] for r in ok),
        **{f'{s}_busy_s_mean': stage(s, 'busy_s') for s in STAGE_NAMES},
        **{f'{s}_active_s_mean': stage(s, 'active_s') for s in STAGE_NAMES},
        'first_statement_s_mean': _mean([r['timing']['first_statement_s'] for r in ok
                                         if r['timing']['first_statement_s'] is not None]),
    }
    return summary


def print_summary(paths: list[Path], common: bool, fmt: str) -> None:
    docids = None
    if common and len(paths) > 1:
        sets = [{k for k, r in latest_records(p).items() if r.get('status') == 'ok'}
                for p in paths]
        docids = set.intersection(*sets)
    summaries = [summarize(p, docids) for p in paths]

    if fmt == 'json':
        print(json.dumps({'common_documents': len(docids) if docids is not None else None,
                          'runs': summaries}, indent=2))
        return

    def f(v: Any, digits: int = 3) -> str:
        if v is None:
            return '-'
        return f'{v:.{digits}f}' if isinstance(v, float) else str(v)

    rows: list[tuple[str, list[str]]] = [
        ('kb / [coreference /] extractor / disambiguator',
         [f'{s["knowledge_base"]}/' + (f'{s["coreference"]}/' if s.get('coreference') else '')
          + f'{s["extractor"]}/{s["disambiguator"]}' for s in summaries]),
        ('documents (ok/err/timeout)',
         [f'{s["ok"]}/{s["error"]}/{s["timeout"]}' for s in summaries]),
        ('extracted / fully linked triples',
         [f'{s["extracted_triples"]}/{s["linked_triples"]}' for s in summaries]),
    ]
    for level in ('strict', 'entity_pair', 'entities'):
        rows.append((f'{level} micro P / R / F1', [
            ' / '.join(f(s[level]['micro'][k]) for k in ('precision', 'recall', 'f1'))
            for s in summaries]))
        rows.append((f'{level} macro P / R / F1', [
            ' / '.join(f(s[level]['macro'][k]) for k in ('precision', 'recall', 'f1'))
            for s in summaries]))
    rows.append(('strict TP / predicted / gold', [
        '/'.join(str(s['strict']['micro'][k]) for k in ('tp', 'predicted', 'gold'))
        for s in summaries]))
    for key, label in [('total_s_mean', 'total s/doc (mean)'),
                       ('total_s_median', 'total s/doc (median)'),
                       ('first_statement_s_mean', 'first statement s (mean)'),
                       ('coreference_busy_s_mean', 'coreference busy s/doc'),
                       ('extraction_busy_s_mean', 'extraction busy s/doc'),
                       ('search_busy_s_mean', 'search busy s/doc'),
                       ('disambiguation_busy_s_mean', 'disambiguation busy s/doc')]:
        rows.append((label, [f(s['timing'][key], 2) for s in summaries]))

    if fmt == 'csv':
        import csv
        w = csv.writer(sys.stdout)
        w.writerow(['metric', *[s['file'] for s in summaries]])
        for label, values in rows:
            w.writerow([label, *values])
        return

    if docids is not None:
        print(f'Compared on the {len(docids)} documents that succeeded in every run '
              '(--all-docs to disable).')
    headers = ['', *[Path(s['file']).name for s in summaries]]
    widths = [max(len(r[0]) for r in rows)] + [
        max(len(headers[i + 1]), *(len(r[1][i]) for r in rows)) for i in range(len(summaries))]
    print('  '.join(h.ljust(w) for h, w in zip(headers, widths)))
    print('  '.join('-' * w for w in widths))
    for label, values in rows:
        print('  '.join(v.ljust(w) for v, w in zip([label, *values], widths)))


def cmd_summary(args: argparse.Namespace) -> int:
    paths = [Path(p) for p in args.files]
    for p in paths:
        if not p.exists():
            raise SystemExit(f'{p}: not found')
    print_summary(paths, common=not args.all_docs, fmt=args.format)
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    records = latest_records(Path(args.file))
    if args.docid is None:
        for r in records.values():
            if args.failed and r.get('status') == 'ok':
                continue
            print(f'{r["docid"]:<12} {_short(r)}  {r.get("title") or ""}')
        return 0
    r = records.get(wikidata_id(args.docid))
    if r is None:
        raise SystemExit(f'{args.docid} not in {args.file}')
    if args.json:
        print(json.dumps(r, indent=2, ensure_ascii=False))
        return 0

    def mark(ok: bool | None) -> str:
        return {True: 'OK ', False: 'XX ', None: '-- '}[ok]

    def part(p: dict[str, Any]) -> str:
        if p['id']:
            return f'{p["id"]} ({p.get("label")})'
        return f'<{p["reason"]}>' if 'reason' in p else f'<{p.get("key")}>'

    t = r['timing']
    print(f'{r["docid"]}  {r.get("title")}  {_short(r)}')
    if r.get('error'):
        print(f'error: {r["error"]["type"]}: {r["error"]["message"]}')
    for level, m in r['metrics'].items():
        print(f'  {level:<12} P={m["precision"]:.3f} R={m["recall"]:.3f} F1={m["f1"]:.3f} '
              f'(tp={m["tp"]}, predicted={m["predicted"]}, gold={m["gold"]})')
    first = t['first_statement_s']
    print(f'timing: total {t["total_s"]:.2f}s, first statement '
          f'{"-" if first is None else f"{first:.2f}s"}, '
          f'split {t["split_s"]:.3f}s')
    for stage in STAGE_NAMES:
        s = t.get(stage)
        if not s or not s['calls']:
            continue
        print(f'  {stage:<15} {s["calls"]:>4} calls  busy {s["busy_s"]:.2f}s  '
              f'active {s["active_s"]:.2f}s  max {s["max_call_s"]:.2f}s  errors {s["errors"]}')

    print('\nextracted triples (OK correct, XX wrong, -- not fully linked):')
    current = None
    for x in r['extracted_triples']:
        if x['sentence'] != current:
            current = x['sentence']
            print(f'  [sentence {current}]')
        print(f'    {mark(x["correct"])}({x["subject"]} | {x["predicate"]} | {x["object"]})')
        print(f'        -> {part(x["linked"]["subject"])}  {part(x["linked"]["predicate"])}  '
              f'{part(x["linked"]["object"])}')
    missed = [g for g in r['gold'] if not g['found']]
    print(f'\ngold triples missed ({len(missed)}/{len(r["gold"])}):')
    for g in missed:
        print(f'    {g["subject"]} {g["property"]} {g["value"]}   {g["surface"]}  '
              f'[sentence {g["sentence"]}]')
    return 0


def rediagnose_record(record: dict[str, Any], doc: dict[str, Any]) -> dict[str, Any]:
    """Recomputes the error diagnosis of a saved record (needs its trace)."""
    gold = {(g['subject'], g['property'], g['value']): g for g in record['gold']}
    candidates: dict[tuple[str, str], list[str]] = {}
    for rec in (record.get('trace') or {}).get('searches', []):
        candidates.setdefault((rec['term'], rec['entity_type']),
                              [c['id'] for c in rec.get('candidates') or []])
    causes = diagnose(record['extracted_triples'], gold, gold_aliases(doc, gold),
                      lambda term, kind: candidates.get((term, kind)))
    for g in record['gold']:
        g['cause'] = causes[(g['subject'], g['property'], g['value'])]
    return record


def cmd_rediagnose(args: argparse.Namespace) -> int:
    """Re-runs the error diagnosis on saved runs (the old files are kept as *.bak)."""
    docs = {doc_key(d): d for d in args.loader(Path(args.dataset))}
    for name in args.files:
        path = Path(name)
        records = list(read_jsonl(path))
        missing = [r['docid'] for r in records if r['docid'] not in docs]
        if missing:
            raise SystemExit(f'{path}: {len(missing)} documents not in {args.dataset}')
        if any('trace' not in r for r in records):
            LOG.warning('%s: records without trace; search misses cannot be told apart', path)
        updated = [rediagnose_record(r, docs[r['docid']]) for r in records]
        with exclusive_lock(path):
            tmp = path.with_name(path.name + '.tmp')
            tmp.write_text(''.join(json.dumps(r, ensure_ascii=False, default=str) + '\n'
                                   for r in updated), encoding='utf-8')
            backup(path)
            tmp.replace(path)
        LOG.info('%s: %d records re-diagnosed', path, len(updated))
    return 0


def cmd_grid(args: argparse.Namespace) -> int:
    """Runs every extractor x disambiguator (x knowledge base) combination.

    Each combination is a separate `run` process (so one failing or crashing
    does not stop the others) writing to its own file in `--output-dir`;
    finished combinations are resumed/skipped like any run.
    """
    out_dir = Path(args.output_dir)
    outputs, failed = [], []
    corefs = [c for c in args.coreferences.split(',')] if args.coreferences else ['']
    combos = [(kb, c, e, d) for kb in args.kbs.split(',') for c in corefs
              for e in args.extractors.split(',') for d in args.disambiguators.split(',')]
    for n, (kb, coreference, extractor, disambiguator) in enumerate(combos, 1):
        name = '_'.join(re.sub(r'[^A-Za-z0-9.-]+', '-', p) for p in (
            args.dataset_name, kb, *([coreference] if coreference else []),
            extractor, disambiguator))
        output = out_dir / f'{name}.jsonl'
        command = [sys.executable, str(args.script), 'run', '--kb', kb,
                   *(['--coreference', coreference] if coreference else []),
                   '--extractor', extractor, '--disambiguator', disambiguator,
                   '-o', str(output), '--no-summary', '--no-progress', *args.run_args]
        LOG.info('[%d/%d] %s + %s%s + %s', n, len(combos), kb,
                 f'{coreference} + ' if coreference else '', extractor, disambiguator)
        start = time.perf_counter()
        code = subprocess.run(command).returncode
        LOG.info('[%d/%d] finished in %.0fs (exit code %d)', n, len(combos),
                 time.perf_counter() - start, code)
        if output.exists():
            outputs.append(output)
        if code:
            failed.append('/'.join(filter(None, (kb, coreference, extractor, disambiguator))))
            if code == 130:
                break
    if outputs:
        print_summary(outputs, common=False, fmt='table')
    if failed:
        LOG.error('Failed or incomplete: %s', ', '.join(failed))
    return 1 if failed else 0


def build_parser(dataset_name: str = 'trex', default_dataset: Path = DEFAULT_DATASET,
                 loader: Any = None, script: str | Path = __file__,
                 description: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description or __doc__.split('\n\n')[0],
        epilog='See the module docstring for examples.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.set_defaults(dataset_name=dataset_name, loader=loader or load_dataset,
                        script=str(script))
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='-v: debug logs of this script, -vv: also of kbel')
    sub = parser.add_subparsers(dest='command', required=True)

    run = sub.add_parser('run', help='evaluate a configuration on the documents',
                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = run.add_argument_group('data')
    g.add_argument('--dataset', default=str(default_dataset), help='dataset JSON file')
    g.add_argument('--offset', type=int, default=0, help='skip the first N selected documents')
    g.add_argument('-n', '--limit', type=int, help='evaluate at most N documents')
    g.add_argument('--docids', help='only these documents: Q1,Q2 or @file')
    g.add_argument('--sample', type=int, help='random sample of N documents (see --seed)')
    g.add_argument('--seed', type=int, default=13)
    g.add_argument('--min-chars', type=int, help='skip shorter documents')
    g.add_argument('--max-chars', type=int, help='skip longer documents')
    g.add_argument('--min-gold', type=int, help='skip documents with fewer gold triples')
    g.add_argument('--max-sentences', type=int,
                   help='use only the first N T-REx sentences of each document (and their gold)')

    g = run.add_argument_group('pipeline')
    g.add_argument('--kb', default='wikidata', help='KIF Search plugin (wikidata, wikidata-vdb, ...)')
    g.add_argument('--extractor', default='mrebel', help='openie, mrebel, relik, relik-cie, kggen, kggen-cluster, llm, ...')
    g.add_argument('--disambiguator', default='naive', help='naive, sim, llm, ...')
    g.add_argument('-K', '--kb-opt', action='append', metavar='KEY=VALUE', help='Search option')
    g.add_argument('-E', '--extractor-opt', action='append', metavar='KEY=VALUE',
                   help='extractor option (repeatable)')
    g.add_argument('-D', '--disambiguator-opt', action='append', metavar='KEY=VALUE',
                   help='disambiguator option (repeatable)')
    g.add_argument('--coreference', metavar='NAME',
                   help='resolve coreferences before anything else (fastcoref, ...); default: none')
    g.add_argument('-C', '--coreference-opt', action='append', metavar='KEY=VALUE',
                   help='coreference option (repeatable)')
    g.add_argument('--plugin', action='append', default=[], metavar='MODULE',
                   help='import MODULE to register custom plugins')
    g.add_argument('--search-limit', type=int, default=10, help='candidates per mention')
    g.add_argument('--language', default='en', help='language of the mentions')
    g.add_argument('--max-concurrency', type=int, default=8,
                   help='concurrent calls in construct (1 = sequential, cleaner stage times)')

    g = run.add_argument_group('LLM plugins (llm, kggen, kggen-cluster)')
    g.add_argument('--llm-model', default=DEFAULT_LLM_MODEL, help='OpenRouter model id')
    g.add_argument('--llm-endpoint', default=OPENROUTER_ENDPOINT)
    g.add_argument('--llm-provider', default='openai',
                   help='build_model provider (OpenRouter is OpenAI-compatible)')
    g.add_argument('--llm-api-key-env', help='env var holding the key '
                   f'(default: {" then ".join(DEFAULT_LLM_KEY_ENVS)})')
    g.add_argument('--llm-temperature', type=float, default=0.0)
    g.add_argument('--env-file', default=str(ROOT / '.env'), help='dotenv file to load')

    g = run.add_argument_group('gold')
    g.add_argument('--gold-annotators', nargs='+', metavar='NAME',
                   help='keep gold triples of these T-REx aligners only '
                        '(Simple-Aligner, SPOAligner, NoSubject-Triple-aligner)')
    g.add_argument('--include-literals', action='store_true',
                   help='keep gold triples with literal objects (never predicted by construct)')

    g = run.add_argument_group('output and safety')
    g.add_argument('-o', '--output', help='JSONL file (default: <output-dir>/<config>.jsonl)')
    g.add_argument('--output-dir', default=str(
        DEFAULT_RUNS_DIR if dataset_name == 'trex' else DEFAULT_RUNS_DIR / dataset_name))
    g.add_argument('--trace', choices=['none', 'compact', 'full'], default='full',
                   help='store every extraction/search/disambiguation call')
    g.add_argument('--timeout', type=float, default=900, help='seconds per document attempt')
    g.add_argument('--retries', type=int, default=2, help='retries of a failed document')
    g.add_argument('--retry-backoff', type=float, default=5.0, help='seconds, doubled each retry')
    g.add_argument('--retry-failed', action='store_true',
                   help='on resume, rerun documents that ended in error/timeout')
    g.add_argument('--max-consecutive-errors', type=int, default=5,
                   help='stop after N failed documents in a row (0 = never)')
    g.add_argument('--overwrite', action='store_true',
                   help='start over; the existing output is renamed to *.bak')
    g.add_argument('--force', action='store_true',
                   help='append even if the configuration differs from the existing output')
    g.add_argument('--dry-run', action='store_true', help='show what would run and exit')
    g.add_argument('--no-progress', action='store_true', help='log lines instead of a progress bar')
    g.add_argument('--no-summary', action='store_true')
    run.set_defaults(func=cmd_run)

    grid = sub.add_parser(
        'grid', help='run every extractor x disambiguator combination',
        description='Options not listed here are forwarded to every `run` '
                    '(e.g. --limit 10 --max-sentences 5 --llm-model ...).')
    grid.add_argument('--extractors', required=True, help='comma-separated, e.g. mrebel,relik,kggen')
    grid.add_argument('--disambiguators', required=True, help='comma-separated, e.g. naive,sim,llm')
    grid.add_argument('--kbs', default='wikidata', help='comma-separated knowledge bases')
    grid.add_argument('--coreferences', default='',
                      help='comma-separated coreference resolvers (default: none); '
                           'use "" in the list for a run without one, e.g. ",fastcoref"')
    grid.add_argument('--output-dir', default=str(DEFAULT_RUNS_DIR / dataset_name))
    grid.set_defaults(func=cmd_grid)

    rediag = sub.add_parser('rediagnose', help='recompute the error diagnosis of saved runs')
    rediag.add_argument('files', nargs='+')
    rediag.add_argument('--dataset', default=str(default_dataset))
    rediag.set_defaults(func=cmd_rediagnose)

    summary = sub.add_parser('summary', help='aggregate and compare runs')
    summary.add_argument('files', nargs='+')
    summary.add_argument('--all-docs', action='store_true',
                         help='do not restrict to documents that succeeded in every run')
    summary.add_argument('--format', choices=['table', 'json', 'csv'], default='table')
    summary.set_defaults(func=cmd_summary)

    show = sub.add_parser('show', help='list the documents of a run, or inspect one')
    show.add_argument('file')
    show.add_argument('docid', nargs='?')
    show.add_argument('--failed', action='store_true', help='list only failed documents')
    show.add_argument('--json', action='store_true', help='print the raw record')
    show.set_defaults(func=cmd_show)
    return parser


def main(argv: list[str] | None = None, **dataset: Any) -> int:
    parser = build_parser(**dataset)
    args, extra = parser.parse_known_args(argv)
    if args.command == 'grid':
        args.run_args = extra
    elif extra:
        parser.error(f'unrecognized arguments: {" ".join(extra)}')
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level if args.verbose > 1 else logging.WARNING,
                        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
                        datefmt='%H:%M:%S')
    LOG.setLevel(level)

    def on_sigterm(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt
    with contextlib.suppress(ValueError):
        signal.signal(signal.SIGTERM, on_sigterm)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
