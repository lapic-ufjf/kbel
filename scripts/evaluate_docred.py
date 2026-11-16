#!/usr/bin/env python3
# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

"""Evaluate KBEL configurations on DocRED (human-annotated triples only).

Uses Linked-DocRED (https://github.com/alteca/Linked-DocRED): the
human-annotated DocRED documents (``dev.json``, ``train_annotated.json``)
whose entities were also linked to Wikidata. The distantly supervised
``train_distant.json`` is refused. A gold triple is
(head Wikidata id, relation Wikidata property, tail Wikidata id); relations
with a numeric or temporal entity, or with an entity that has no Wikidata
id, are left out (kept as literals with ``--include-literals``).

Same commands, options, output format and safety as ``evaluate_trex.py``
(resume, config check, per-stage timing, traces), plus ``download``::

    python scripts/evaluate_docred.py download
    python scripts/evaluate_docred.py run --limit 10 --extractor relik --disambiguator sim
    python scripts/evaluate_docred.py grid --limit 10 \\
        --extractors openie,mrebel,relik,kggen,llm --disambiguators naive,sim,llm
    python scripts/evaluate_docred.py summary evaluation/runs/docred/*.jsonl
    python scripts/evaluate_docred.py show evaluation/runs/docred/<run>.jsonl dev-0000

Results are explored in ``evaluation/docred_results.ipynb``.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'evaluation' / 'dataset' / 'docred'
DEFAULT_DATASET = DATA_DIR / 'dev.json'
SOURCE = 'https://raw.githubusercontent.com/alteca/Linked-DocRED/main/Linked-DocRED'
HUMAN_SPLITS = ('dev.json', 'train_annotated.json')
FILES = (*HUMAN_SPLITS, 'rel_info.json', 'README.md')

WD = 'http://www.wikidata.org/entity/'
WDT = 'http://www.wikidata.org/prop/direct/'
LITERAL_TYPES = {'NUM', 'TIME'}


def _load_core() -> Any:
    name = 'evaluate_trex'
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f'{name}.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


core = _load_core()

_NO_SPACE_BEFORE = re.compile(r"\s+([,.;:!?%)\]}]|'s\b|n't\b|'re\b|'ve\b|'ll\b|'d\b|'m\b)")
_NO_SPACE_AFTER = re.compile(r'([(\[{$£€])\s+')
_HYPHEN = re.compile(r'(?<=\w)\s+([-–/])\s+(?=\w)')
_QUOTED = re.compile(r'"\s+([^"]*?)\s+"')


def detokenize(tokens: list[str]) -> str:
    """Joins DocRED tokens into natural text ("free - to - air" -> "free-to-air")."""
    text = ' '.join(t for t in tokens if t.strip())
    text = text.replace('``', '"').replace("''", '"').replace('-LRB-', '(').replace('-RRB-', ')')
    text = _HYPHEN.sub(r'\1', text)
    text = _NO_SPACE_BEFORE.sub(r'\1', text)
    text = _NO_SPACE_AFTER.sub(r'\1', text)
    return _QUOTED.sub(r'"\1"', text)


XSD = 'http://www.w3.org/2001/XMLSchema#'


def _literal_uri(entity_type: str | None, name: str) -> str:
    """The gold URI of a numeric or temporal entity, as T-REx writes literals.

    Dates become ``YYYY-MM-DDT00:00:00Z^^xsd:dateTime`` and numbers
    ``N^^xsd:decimal``; whatever cannot be parsed stays ``literal:<name>``.
    """
    from kbel.core.values import parse_quantity, parse_time
    if entity_type == 'TIME':
        value = parse_time(name)
        if value is not None:
            return f'{value.time.date().isoformat()}T00:00:00Z^^{XSD}dateTime'
    elif entity_type == 'NUM':
        parsed = parse_quantity(name)
        if parsed is not None and parsed[1] is None:
            return f'{parsed[0]}^^{XSD}decimal'
    return f'literal:{name}'


def _entity(entity: dict[str, Any]) -> tuple[str | None, str]:
    """(IRI or None when the entity cannot be a Wikidata item, surface form)."""
    name = max((m['name'] for m in entity['mentions']), key=len, default='')
    qid = (entity.get('entity_linking') or {}).get('wikidata_resource')
    if entity.get('type') in LITERAL_TYPES or not qid or not str(qid).startswith('Q'):
        return None, name
    return WD + qid, name


def convert(doc: dict[str, Any], docid: str) -> dict[str, Any]:
    """A Linked-DocRED document in the T-REx layout used by the evaluator."""
    text, bounds = '', []
    for tokens in doc['sents']:
        sentence = detokenize(tokens)
        start = len(text) + (1 if text else 0)
        text = f'{text} {sentence}' if text else sentence
        bounds.append([start, len(text)])

    entities = {e['id']: e for e in doc['entities']}
    triples = []
    for relation in doc['relations']:
        head, tail = entities[relation['h']], entities[relation['t']]
        (head_iri, head_name), (tail_iri, tail_name) = _entity(head), _entity(tail)
        tail_uri = tail_iri or _literal_uri(tail.get('type'), tail_name)
        sentences = relation.get('evidence') or [
            max(min(m['sent_id'] for m in e['mentions']) for e in (head, tail))]
        triples.append({
            'sentence_id': max(sentences),
            'annotator': 'human',
            'subject': {'uri': head_iri or f'literal:{head_name}', 'surfaceform': head_name},
            'predicate': {'uri': WDT + relation['r'], 'surfaceform': None},
            'object': {'uri': tail_uri, 'surfaceform': tail_name},
        })
    triples = [t for t in triples if t['subject']['uri'].startswith(WD)]
    mentions = [{'surfaceform': m['name'], 'uri': iri}
                for e in doc['entities'] for iri in [_entity(e)[0]] if iri
                for m in e['mentions']]
    return {'docid': docid, 'uri': None, 'title': doc.get('title'), 'text': text,
            'sentences_boundaries': bounds, 'triples': triples, 'entities': mentions}


def load_dataset(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if 'distant' in path.name:
        raise SystemExit(f'{path.name} is distantly supervised; use the human-annotated '
                         f'splits ({", ".join(HUMAN_SPLITS)}).')
    if not path.exists():
        raise SystemExit(f'{path} not found. Download it with: '
                         f'python {Path(__file__).name} download')
    docs = json.loads(path.read_text(encoding='utf-8'))
    if not docs or 'entities' not in docs[0] or 'relations' not in docs[0]:
        raise SystemExit(f'{path}: not a Linked-DocRED file (expected `entities` with '
                         '`entity_linking` and `relations`).')
    split = path.stem.split('_')[0]
    return [convert(doc, f'{split}-{i:04d}') for i, doc in enumerate(docs)]


def download(files: tuple[str, ...] = ('dev.json', 'rel_info.json', 'README.md'),
             target: Path = DATA_DIR, force: bool = False) -> int:
    target.mkdir(parents=True, exist_ok=True)
    for name in files:
        path = target / name
        if path.exists() and not force:
            print(f'{path} already there')
            continue
        print(f'Downloading {SOURCE}/{name} ...')
        tmp = path.with_name(path.name + '.part')
        urllib.request.urlretrieve(f'{SOURCE}/{name}', tmp)
        tmp.replace(path)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ['download']:
        import argparse
        parser = argparse.ArgumentParser(prog='evaluate_docred.py download')
        parser.add_argument('--train', action='store_true',
                            help='also download train_annotated.json (24 MB)')
        parser.add_argument('--force', action='store_true')
        args = parser.parse_args(argv[1:])
        files = ('dev.json', 'rel_info.json', 'README.md') + (
            ('train_annotated.json',) if args.train else ())
        return download(files, force=args.force)
    return core.main(argv, dataset_name='docred', default_dataset=DEFAULT_DATASET,
                     loader=load_dataset, script=__file__, description=__doc__.split('\n\n')[0])


if __name__ == '__main__':
    sys.exit(main())
