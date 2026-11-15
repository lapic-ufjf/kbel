# KBEL — Knowledge Base Entity Linking

KBEL is a Python framework that links text to entities in a knowledge base
(Wikidata, DBpedia, …). It can work at two levels:

- **Mention linking**: given a term (e.g. *"Python"*) and its sentence, find
  the knowledge-base entity it refers to (`wd:Q28865`, the programming
  language, not the snake).
- **Text → knowledge graph**: extract subject–predicate–object triples from
  free text and link each part to an item or property.

Every stage is a plugin chosen by name, so extraction, search and
disambiguation strategies can be combined freely.

```
                 ┌──────────────┐  triples   ┌──────────┐ candidates ┌────────────────┐
  text  ───────► │  Extractor   │ ─────────► │  Search  │ ─────────► │ Disambiguator  │ ──► linked entities
                 │ openie·mrebel│  mentions  │ wikidata │            │ naive·sim·llm  │
                 │     ·llm     │            │ dbpedia… │            │                │
                 └──────────────┘            └──────────┘            └────────────────┘
```

See [`examples/demo.ipynb`](./examples/demo.ipynb) for a guided tour that
combines all of them.

---

## Installation

KBEL requires Python 3.11–3.13.

```bash
pip install "kbel @ git+https://github.com/lapic-ufjf/kbel.git"
```

Optional features are installed as extras:

| Extra | Enables | Command |
|-------|---------|---------|
| *(none)* | `naive` disambiguator, all knowledge bases | `pip install kbel` |
| `similarity` | `sim` disambiguator (sentence embeddings) | `pip install "kbel[similarity]"` |
| `llm` | `llm` disambiguator and extractor (OpenAI, Ollama, watsonx) | `pip install "kbel[llm]"` |
| `mrebel` | `mrebel` extractor (PyTorch + Transformers) | `pip install "kbel[mrebel]"` |
| `openie` | `openie` extractor — needs only **Java**; CoreNLP is downloaded on first use | `pip install "kbel[openie]"` |
| `kggen` | `kggen` and `kggen-cluster` extractors ([KGGen](https://github.com/stair-lab/kg-gen), any LiteLLM model) | `pip install "kbel[kggen]"` |
| `relik` | `relik` and `relik-cie` extractors — need a **Python 3.10/3.11** interpreter; ReLiK is installed in its own environment (`~/.kbel/relik-venv`) on first use | `pip install "kbel[relik]"` |
| `dev` | tests and tooling | `pip install -e ".[dev]"` |

Missing optional dependencies raise an `ImportError` telling you which extra
to install.

---

## Quick start

```python
from kbel.core.kbel import KBEL
from kbel.core.mention import Mention, EntityType

from kbel.disambiguators import naive     # registers the "naive" plugin

kbel = KBEL('wikidata', disambiguator='naive')

mention = Mention(
    term='Python',
    text='Python is a programming language used for coding.',
    entity_type=EntityType.ITEM,
)

for result in kbel.link(mention):
    print(result.entity, result.candidate.label, '-', result.candidate.description)
# Item(IRI('http://www.wikidata.org/entity/Q28865')) Python - general-purpose programming language
```

And from raw text to linked triples:

```python
from kbel.extractors import mrebel        # registers the "mrebel" plugin

kbel.extractor = 'mrebel'

for triple, (subjects, predicates, objects) in kbel.extract_and_link(
        'Ada Lovelace worked with Charles Babbage on the Analytical Engine.', limit=1):
    print(triple, subjects, predicates, objects, sep='\n  ')
```

To build knowledge-base statements from a whole document, use `construct`.
It splits the text into paragraphs and sentences, extracts triples from each
sentence (in batches, for extractors that support them, such as mREBEL),
links subject (item), predicate (property) and object (item) — with the
sentence as the mention's `text` and its paragraph as `context`, which the
`llm` disambiguator includes in its prompt — and
yields a [KIF](https://github.com/PUC-Behring-AI/kif) `Statement` for each
triple whose three parts were linked:

```python
for statement in kbel.construct(document, language='en'):
    print(statement.subject, statement.snak.property, statement.snak.value)
# Statement(Item(wd:Q7186), ValueSnak(Property(wd:P19), Item(wd:Q270))) — Marie Curie, place of birth, Warsaw

statements = list(kbel.construct(document))        # all at once

# In async code (or Jupyter), iterate the async version:
async for statement in kbel.aconstruct(document, max_concurrency=8):
    ...
```

`construct` is lazy: statements come out in text order, each as soon as its
triple is linked, while the rest of the document is processed concurrently
(work for earlier sentences is prioritized). Stopping the iteration cancels
the remaining work. Mentions with the same term share one search, and
repeated statements are yielded once.

Every `KBEL` method has an async version — `acandidates_lookup`, `alink`,
`aextract`, `aextract_and_link`, `aconstruct` — and so do the plugins
(`TripleExtractor.aextract`, `Disambiguator.adisambiguate`). The LLM plugins
call the model asynchronously; other plugins run in worker threads, so the
event loop is never blocked. The sync `extract_and_link` and `construct`
use the async machinery internally, so they also link concurrently.

> **Plugins register on import.** Strategies are looked up by name, so the
> module that defines a plugin must be imported once before its name is used
> (`from kbel.disambiguators import similarity`, `from kbel.extractors import openie`, …).
> Prefer this `from … import` form: `import kbel.extractors.openie` would
> rebind the name `kbel` to the package, hiding a `kbel = KBEL(...)` instance.
> Knowledge bases (`Search` plugins) are registered automatically.

---

## Core concepts

| Object | Module | Role |
|--------|--------|------|
| `KBEL` | `kbel.core.kbel` | Orchestrates the pipeline: `candidates_lookup`, `link`, `extract`, `extract_and_link`, `construct` — each with an async `a…` version. |
| `Mention` | `kbel.core.mention` | What to link: `term`, the sentence `text`, optional `context`, `entity_type` (`ITEM`/`PROPERTY`) and `language`. |
| `Candidate` | `kbel.core.candidate` | An entity returned by the search: `iri`, `label`, `description`. |
| `LinkResult` | `kbel.core.results` | A linked entity: KIF `entity` (`Item`/`Property`), its `candidate` and a `score`. |
| `Triple` | `kbel.core.triple` | `subject`, `predicate`, `object` strings extracted from text. |

The three pluggable stages are set by name (or by passing an instance) either
in the constructor or later:

```python
kbel = KBEL('wikidata-vdb', disambiguator='sim', extractor='openie', search_limit=20)

kbel.knowledge_base = 'wikidata'                    # switch search backend
kbel.disambiguator = Disambiguator('sim', model_name='Snowflake/snowflake-arctic-embed-xs', context_weight=1.0)
kbel.extractor = TripleExtractor('mrebel', num_beams=6)
```

### Knowledge bases (`Search`)

Search backends come from [KIF](https://github.com/PUC-Behring-Institute-for-AI/kif)
plus the ones shipped with KBEL. Any name in `kif_lib.search.Search.registry`
works; the most useful are:

| Name | Backend |
|------|---------|
| `wikidata` | Wikidata API full-text search (fast, label based) |
| `wikidata-vdb` | [Wikidata Vector Database](https://wd-vectordb.wmcloud.org) semantic search (provided by KBEL) |
| `wikidata-ddgs` | Wikidata pages found through DuckDuckGo |
| `dbpedia` | DBpedia Lookup |
| `pubchem` | PubChem (chemistry) |

### Disambiguators

| Name | Strategy | Extra |
|------|----------|-------|
| `naive` | Returns the candidate whose label matches the term, else the first one. | — |
| `sim` | Scores each candidate (`label: description`) against the sentence **and its paragraph** with sentence embeddings, plus a prior on its search rank and an exact-label bonus. Default model `BAAI/bge-small-en-v1.5`; lighter: `Snowflake/snowflake-arctic-embed-xs`, `sentence-transformers/static-retrieval-mrl-en-v1`. | `similarity` |
| `llm` | Asks an LLM to choose among the candidates, using `text` and `context`. | `llm` |

```python
from kbel.disambiguators import Disambiguator, llm   # llm registers the plugin
from langchain_openai import ChatOpenAI

kbel.disambiguator = Disambiguator('llm', model=ChatOpenAI(model='gpt-5.2'))
# or let KBEL build the model:
kbel.disambiguator = Disambiguator(
    'llm', model_name='llama3.1', model_provider='ollama',
    model_endpoint='http://localhost:11434', model_apikey='-')
```

### Triple extractors

| Name | Approach | Predicates | Notes |
|------|----------|------------|-------|
| `openie` | Stanford OpenIE (rule based, CoreNLP) | phrases from the text (`was born in`) | English only; needs Java. CoreNLP is located, downloaded (~500 MB) and started automatically. Pronouns are resolved (*He* → *Alan Turing*). |
| `mrebel` | [mREBEL](https://github.com/Babelscape/rebel) seq2seq model | Wikidata property labels (`place of birth`) | 18 languages (`source_language='pt_XX'`, …). Model weights are CC BY-NC-SA 4.0 (non-commercial). |
| `relik` | [ReLiK](https://github.com/SapienzaNLP/relik) relation extraction (retriever + reader) | Wikidata property labels (`place of birth`) | Subject/object as written in the text. Default `relik-ie/relik-relation-extraction-large` (`-small` is lighter). |
| `relik-cie` | ReLiK closed IE: entity linking + relation extraction | Wikidata property labels | Subject/object are the **linked Wikipedia titles** (*NBA* → *National Basketball Association*). Its entity index is large (~8 GB, loaded in memory). |
| `kggen` | [KGGen](https://github.com/stair-lab/kg-gen): an LLM extracts the entities, then relations among them (DSPy) | free text | Any LiteLLM model; default `openrouter/openai/gpt-4o-mini`. Sentences of a batch are sent concurrently. |
| `kggen-cluster` | KGGen + clustering of equivalent entities across the batch | free text | *Turing* and *Alan Turing* become one name. Predicates are only clustered with `cluster_edges=True` (small models over-merge them). |
| `llm` | Prompted LLM | free text | Custom prompt via `extract(text, prompt=...)`. |

Useful options:

```python
TripleExtractor('mrebel', source_language='pt_XX', num_beams=6, split_sentences=True)
TripleExtractor('openie', corenlp_home='~/Downloads/stanford-corenlp-4.6.0.zip')
TripleExtractor('openie', resolve_coref=False)      # faster, keeps pronouns
TripleExtractor('openie', remove_parentheticals=False)
TripleExtractor('relik', model_name='relik-ie/relik-relation-extraction-small', min_confidence=0.5)
TripleExtractor('relik-cie', use_entity_titles=True, keep_unlinked=False)
TripleExtractor('kggen', model='openrouter/openai/gpt-4o-mini')     # OPENROUTER_API_KEY
TripleExtractor('kggen-cluster', model='ollama_chat/llama3.1', api_base='http://localhost:11434',
                batch_size=16, context='biographies')
```

ReLiK pins torch 2.3, which has no build for recent Pythons, so KBEL runs
it in a separate process with its own interpreter: `~/.kbel/relik-venv`
(created on first use from a Python 3.10/3.11 found on the system), or the
one given by `python=` or the `KBEL_RELIK_PYTHON` variable.

Any extractor can render its triples as a graph (requires
[GraphViz](https://graphviz.org)):

```python
kbel.extract(text, graph=True, graph_path='graph.png')
```

---

## Extending KBEL

Subclass a base class and give it a `strategy_name`; it becomes available by
name as soon as the module is imported.

```python
from kbel.disambiguators import Disambiguator
from kbel.core.results import DisambiguationResult

class LongestDescription(Disambiguator, strategy_name='longest'):
    def _disambiguate(self, mention, candidates, limit, *args, **kwargs):
        ranked = sorted(candidates, key=lambda c: len(c.description or ''), reverse=True)
        return [DisambiguationResult(candidate=c) for c in ranked[:limit]]

kbel.disambiguator = 'longest'
```

```python
from kbel.extractors import TripleExtractor
from kbel.core.triple import Triple

class IsA(TripleExtractor, strategy_name='is-a'):
    def __init__(self, strategy_name, **kwargs):
        super().__init__(**kwargs)

    def _extract(self, text, prompt=None):
        subject, _, rest = text.partition(' is a ')
        return [Triple(subject.strip(), 'instance of', rest.strip(' .'))] if rest else []
```

The async versions (`aextract`, `adisambiguate`) work out of the box: by
default they run `_extract` / `_disambiguate` in a worker thread, so these
must be thread-safe (guard shared state, such as a tokenizer, with a lock).
An extractor that is faster on several texts at once (e.g. a batched model)
overrides `_extract_batch(texts)` and sets `batch_size`; `construct` then
sends it that many sentences per call. A plugin with a natively async
backend overrides the async hook instead:

```python
class RemoteNER(TripleExtractor, strategy_name='remote'):
    def _extract(self, text, prompt=None):
        return httpx.post(URL, json={'text': text}).json()['triples']

    async def _aextract(self, text, prompt=None):
        async with httpx.AsyncClient() as client:
            return (await client.post(URL, json={'text': text})).json()['triples']
```

New knowledge bases are KIF `Search` plugins
(see [`kbel/search/wikidata_vdb.py`](./src/kbel/search/wikidata_vdb.py)).

---

## Development

```bash
pip install -e ".[dev,similarity,llm,mrebel]"
make test            # run the test suite (CoreNLP integration tests are skipped without Java)
make license-check   # verify the license header of every source file
make license-add     # add the header where it is missing
```

### Evaluating on T-REx

`scripts/evaluate_trex.py` runs `construct` on T-REx documents
(`evaluation/dataset/re-nlg_0-10000.json`) and compares the statements with
the gold Wikidata triples. Each document becomes one JSONL line with the
extracted triples (and how each part was linked), precision/recall/F1, the
time of every stage and the full trace of extraction, search and
disambiguation calls. Runs resume where they stopped, and appending results
of a different configuration is refused.

```bash
python scripts/evaluate_trex.py run --limit 50 --kb wikidata --extractor mrebel --disambiguator sim
python scripts/evaluate_trex.py run --limit 50 --extractor kggen --disambiguator llm # OpenRouter (OPENROUTER_API_KEY)
python scripts/evaluate_trex.py summary evaluation/runs/*.jsonl                      # compare runs
python scripts/evaluate_trex.py show evaluation/runs/<run>.jsonl Q33199              # inspect a document
```

Plugin options are passed as `-E key=value` (extractor), `-D key=value`
(disambiguator) and `-K key=value` (knowledge base); see `run --help`.

---

## License

KBEL is released under the [Apache-2.0 license](./LICENSE).

Third-party components keep their own licenses: the mREBEL model weights
are distributed by Babelscape under CC BY-NC-SA 4.0 (non-commercial), ReLiK
is distributed by SapienzaNLP under CC BY-NC-SA 4.0 (non-commercial), and
Stanford CoreNLP, run as a separate process by the `openie` extractor, is
GPL-3.0.
