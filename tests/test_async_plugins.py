# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

"""Native async implementations of the LLM plugins and the wikidata-vdb search."""

import asyncio

import pytest

from kbel.core.candidate import Candidate
from kbel.core.mention import Mention
from kbel.core.triple import Triple
from kbel.disambiguators import Disambiguator
from kbel.extractors import TripleExtractor

langchain_core = pytest.importorskip('langchain_core')

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

from kbel.disambiguators import llm  # noqa: E402,F401  (registers "llm")
from kbel.extractors.llm import llm_extractor  # noqa: E402,F401


class ScriptedChatModel(BaseChatModel):
    """Answers `reply`; `async_only` makes the sync path fail."""

    reply: str
    async_only: bool = False
    sync_calls: int = 0
    async_calls: int = 0
    prompts: list = []

    @property
    def _llm_type(self) -> str:
        return 'scripted'

    def _result(self) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.sync_calls += 1
        self.prompts.append(messages[-1].content)
        if self.async_only:
            raise AssertionError('sync path used')
        return self._result()

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        self.async_calls += 1
        self.prompts.append(messages[-1].content)
        await asyncio.sleep(0)
        return self._result()


CANDIDATES = [
    Candidate(iri='http://www.wikidata.org/entity/Q271218', id='Q271218',
              label='Python', description='genus of reptiles'),
    Candidate(iri='http://www.wikidata.org/entity/Q28865', id='Q28865',
              label='Python', description='general-purpose programming language'),
]
MENTION = Mention(term='Python', text='Python is a programming language.',
                  context='A paragraph about software.')
TRIPLES_REPLY = ('subject: Python | predicate: created by | object: Guido van Rossum\n'
                 'subject: Python | predicate: instance of | object: programming language')


# -- LLM disambiguator ----------------------------------------------------------

def test_llm_disambiguator_sync_and_async_agree():
    model = ScriptedChatModel(reply='Q28865')
    d = Disambiguator('llm', model=model)

    sync = d.disambiguate(MENTION, CANDIDATES)
    async_ = asyncio.run(d.adisambiguate(MENTION, CANDIDATES))

    assert [r.candidate.id for r in sync] == [r.candidate.id for r in async_] == ['Q28865']
    assert (model.sync_calls, model.async_calls) == (1, 1)


def test_llm_disambiguator_async_is_native():
    model = ScriptedChatModel(reply='Q28865, Q271218', async_only=True)
    results = asyncio.run(Disambiguator('llm', model=model).adisambiguate(
        MENTION, CANDIDATES, limit=2))

    assert [r.candidate.id for r in results] == ['Q28865', 'Q271218']
    assert model.sync_calls == 0


def test_llm_prompt_includes_the_context():
    model = ScriptedChatModel(reply='Q28865')
    Disambiguator('llm', model=model).disambiguate(MENTION, CANDIDATES)
    [prompt] = model.prompts
    assert 'Sentence: "Python is a programming language."' in prompt
    assert 'Context: "A paragraph about software."' in prompt


@pytest.mark.parametrize('context', [None, '', MENTION.text])
def test_llm_prompt_omits_missing_or_redundant_context(context):
    model = ScriptedChatModel(reply='Q28865')
    mention = Mention(term=MENTION.term, text=MENTION.text, context=context)
    Disambiguator('llm', model=model).disambiguate(mention, CANDIDATES)
    assert 'Context:' not in model.prompts[0]


def test_construct_gives_the_llm_the_paragraph(monkeypatch):
    from kbel.core.kbel import KBEL

    class OneTriple(TripleExtractor, strategy_name='test-llm-context-extract'):
        def __init__(self, strategy_name):
            super().__init__()

        def _extract(self, text, prompt=None):
            return [Triple('Python', 'instance of', 'Python')] if 'It' in text else []

    monkeypatch.setattr(KBEL, 'candidates_lookup', lambda self, m, limit=None: list(CANDIDATES))
    model = ScriptedChatModel(reply='Q28865', async_only=True)
    kbel = KBEL('empty', disambiguator=Disambiguator('llm', model=model),
                extractor=TripleExtractor('test-llm-context-extract'))
    paragraph = 'Python was created by Guido van Rossum. It was released in 1991.'
    list(kbel.construct(paragraph + '\n\nA second paragraph.'))

    assert model.prompts
    for prompt in model.prompts:
        assert 'Sentence: "It was released in 1991."' in prompt
        assert f'Context: "{paragraph}"' in prompt


def test_llm_disambiguator_concurrent_calls():
    model = ScriptedChatModel(reply='Q28865', async_only=True)
    d = Disambiguator('llm', model=model)

    async def many():
        return await asyncio.gather(*(d.adisambiguate(MENTION, CANDIDATES) for _ in range(5)))

    assert len(asyncio.run(many())) == 5
    assert model.async_calls == 5


def test_llm_disambiguator_unknown_ids_raise():
    model = ScriptedChatModel(reply='Q999', async_only=True)
    with pytest.raises(ValueError, match='Could not disambiguate'):
        asyncio.run(Disambiguator('llm', model=model).adisambiguate(MENTION, CANDIDATES))


def test_llm_prompt_numbers_the_candidates_instead_of_showing_iris():
    model = ScriptedChatModel(reply='C2')
    [result] = Disambiguator('llm', model=model).disambiguate(MENTION, CANDIDATES)
    assert result.candidate.id == 'Q28865'
    [prompt] = model.prompts
    assert 'ID: C1\n' in prompt and 'ID: C2\n' in prompt
    assert 'wikidata.org' not in prompt and 'Q28865' not in prompt


@pytest.mark.parametrize('reply', [
    'C2', 'c2', '`C2`', 'Q28865', 'http://www.wikidata.org/entity/Q28865', 'C2, C2', 'C9, C2'])
def test_llm_disambiguator_accepts_the_usual_ways_of_naming_a_candidate(reply):
    iri_only = [Candidate(iri=c.iri, id=c.iri, label=c.label, description=c.description)
                for c in CANDIDATES]
    for candidates in (CANDIDATES, iri_only):
        results = Disambiguator('llm', model=ScriptedChatModel(reply=reply)).disambiguate(
            MENTION, candidates)
        assert [r.candidate.iri for r in results] == ['http://www.wikidata.org/entity/Q28865']


@pytest.mark.parametrize('reply', ['', 'NONE', 'none.'])
def test_llm_disambiguator_without_a_match_returns_nothing(reply):
    model = ScriptedChatModel(reply=reply, async_only=True)
    assert asyncio.run(Disambiguator('llm', model=model).adisambiguate(MENTION, CANDIDATES)) == []


def test_llm_disambiguator_keeps_one_chain_per_instance():
    first = Disambiguator('llm', model=ScriptedChatModel(reply='C1'))
    second = Disambiguator('llm', model=ScriptedChatModel(reply='C2'))
    assert first.disambiguate(MENTION, CANDIDATES)[0].candidate.id == 'Q271218'
    assert second.disambiguate(MENTION, CANDIDATES)[0].candidate.id == 'Q28865'
    assert first._chain(True) is first._chain(True)
    assert first._chain(True) is not second._chain(True)


def test_build_model_validates_its_arguments():
    from kbel.disambiguators.llm.utils import build_model

    with pytest.raises(ValueError, match='model_apikey'):
        build_model('gpt-4o-mini', 'openai', 'https://api.openai.com/v1', None)
    with pytest.raises(ValueError, match='provider'):
        build_model('x', 'nope', 'http://localhost', 'key')
    with pytest.raises(ValueError, match='project_id'):
        build_model('x', 'ibm', 'http://localhost', 'key')
    with pytest.raises(ValueError, match='model_name'):
        Disambiguator('llm')


def test_build_model_passes_parameters_to_ollama_and_needs_no_key(monkeypatch):
    import sys
    import types

    from kbel.disambiguators.llm.utils import build_model

    created = {}
    module = types.ModuleType('langchain_ollama')
    module.ChatOllama = lambda **kwargs: created.update(kwargs) or 'model'
    monkeypatch.setitem(sys.modules, 'langchain_ollama', module)
    assert build_model('llama3.1', 'ollama', 'http://localhost:11434', temperature=0.2) == 'model'
    assert created == {'model': 'llama3.1', 'base_url': 'http://localhost:11434',
                       'temperature': 0.2}


def test_llm_disambiguator_requires_text():
    model = ScriptedChatModel(reply='Q28865')
    with pytest.raises(AssertionError):
        asyncio.run(Disambiguator('llm', model=model).adisambiguate(
            Mention(term='Python', text=''), CANDIDATES))


# -- LLM extractor -----------------------------------------------------------------

def test_llm_extractor_sync_and_async_agree():
    model = ScriptedChatModel(reply=TRIPLES_REPLY)
    e = TripleExtractor('llm', model=model)

    expected = [Triple('Python', 'created by', 'Guido van Rossum'),
                Triple('Python', 'instance of', 'programming language')]
    assert e.extract('Python was created by Guido van Rossum.') == expected
    assert asyncio.run(e.aextract('Python was created by Guido van Rossum.')) == expected


def test_llm_extractor_async_is_native():
    model = ScriptedChatModel(reply=TRIPLES_REPLY, async_only=True)
    triples = asyncio.run(TripleExtractor('llm', model=model).aextract(
        'Python was created by Guido van Rossum.', prompt='Custom system prompt.'))
    assert len(triples) == 2
    assert model.sync_calls == 0


def test_llm_extractor_custom_prompt_may_contain_braces_and_ask_for_json():
    reply = '```json\n[{"subject": "Python", "predicate": "created by", "object": "Guido"}]\n```'
    model = ScriptedChatModel(reply=reply)
    prompt = 'Answer with JSON: [{"subject": "...", "predicate": "...", "object": "..."}]'
    extractor = TripleExtractor('llm', model=model)
    assert extractor.extract('Python {was} created by Guido.', prompt=prompt) == [
        Triple('Python', 'created by', 'Guido')]
    assert 'Text: "Python {was} created by Guido."' in model.prompts[0]


def test_llm_extractor_ignores_bullets_and_incomplete_lines():
    reply = ('1. subject: A | predicate: b | object: C\n'
             '- subject: D | predicate: e: f | object: G\n'
             'subject: H | predicate: i | object:\n'
             'Here are the triples')
    extractor = TripleExtractor('llm', model=ScriptedChatModel(reply=reply))
    assert extractor.extract('Some text.') == [Triple('A', 'b', 'C'), Triple('D', 'e: f', 'G')]


def test_default_extraction_prompt_does_not_force_person_properties():
    from kbel.extractors.llm.constants import TE_DEFAULT_PROMPT

    assert '"country of citizenship" is only for people' in TE_DEFAULT_PROMPT
    assert 'keep the pronoun' in TE_DEFAULT_PROMPT
    assert '`unknown value`' in TE_DEFAULT_PROMPT and '`no value`' in TE_DEFAULT_PROMPT


# -- wikidata-vdb search -------------------------------------------------------------

VDB_ITEMS = [{'QID': 'Q270', 'similarity_score': 0.9}, {'QID': 'Q992560'}, {'other': 1}]
ENTITIES = {'entities': {
    'Q270': {'id': 'Q270', 'labels': {'en': {'value': 'Warsaw'}},
             'descriptions': {'en': {'value': 'capital of Poland'}}},
    'Q992560': {'id': 'Q992560', 'missing': ''},
}}


@pytest.fixture
def vdb(monkeypatch):
    from kif_lib.search import Search
    from kbel.search import WikidataVDB_Search

    search = Search('wikidata-vdb', limit=5)
    calls = {'sync': [], 'async': []}

    def reply(iri, params):
        return ENTITIES if iri == WikidataVDB_Search.WIKIDATA_ENDPOINT else VDB_ITEMS

    def get_json(iri, params=None, timeout=None):
        calls['sync'].append(iri)
        return reply(iri, params)

    async def aget_json(iri, params=None, timeout=None):
        calls['async'].append(iri)
        return reply(iri, params)

    monkeypatch.setattr(search, '_http_get_json', get_json)
    monkeypatch.setattr(search, '_http_aget_json', aget_json)
    return search, calls


def _labels(pairs):
    return [(e.iri.content.rsplit('/', 1)[-1], d['labels']['en'].content) for e, d in pairs]


def test_vdb_async_search_matches_sync(vdb):
    search, calls = vdb

    async def collect():
        return [x async for x in search.aitem_descriptor(search='Warsaw', limit=5)]

    async_results = _labels(asyncio.run(collect()))
    sync_results = _labels(search.item_descriptor(search='Warsaw', limit=5))

    assert async_results == sync_results == [('Q270', 'Warsaw')]   # missing/invalid skipped
    assert calls['async'][0].endswith('/item/query/')               # vector DB, not wbsearchentities
    assert len(calls['async']) == len(calls['sync']) == 2


def test_vdb_entities_use_the_canonical_wikidata_iris(vdb):
    search, _ = vdb
    [(item, _)] = search.item_descriptor(search='Warsaw', limit=5)
    assert item.iri.content == 'http://www.wikidata.org/entity/Q270'


def test_vdb_does_not_download_the_claims_of_the_results():
    from kif_lib.search import Search

    [params] = Search('wikidata-vdb')._entities_params(['Q270', 'P19'], 'en')
    assert params['ids'] == 'Q270|P19'
    assert set(params['props'].split('|')) == {'labels', 'descriptions', 'aliases', 'datatype'}


def test_vdb_async_property_search_uses_property_endpoint(vdb):
    search, calls = vdb

    async def collect():
        return [x async for x in search.aproperty_descriptor(search='born in', limit=5)]

    asyncio.run(collect())
    assert calls['async'][0].endswith('/property/query/')


def test_vdb_async_client_follows_the_event_loop():
    from kif_lib.search import Search

    search = Search('wikidata-vdb')

    async def client():
        return search.get_aclient()

    async def same_loop():
        return search.get_aclient() is search.get_aclient()

    first, second = asyncio.run(client()), asyncio.run(client())
    assert first is not second          # a new loop gets a new client
    assert asyncio.run(same_loop())     # reused within one loop
