# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

import shutil
import socket
import zipfile

import pytest

from kbel.core.triple import Triple
from kbel.extractors import TripleExtractor
from kbel.extractors.openie import CoreNLPServer, OpenIETripleExtractor
from kbel.extractors.openie import corenlp
from kbel.extractors.openie.openie_extractor import CITATION_RE


class FakeServer:
    """Stands in for `CoreNLPServer`: records requests, returns `response`."""

    def __init__(self, response=None):
        self.response = response or {'sentences': []}
        self.started = self.closed = False
        self.requests = []

    def start(self):
        self.started = True
        return self

    def close(self):
        self.closed = True

    def annotate(self, text, properties):
        self.requests.append((text, properties))
        return self.response


def openie(*triples):
    """CoreNLP JSON with one sentence holding the given OpenIE triples."""
    return {'sentences': [{'openie': [
        {'subject': s, 'relation': p, 'object': o} for s, p, o in triples]}]}


def as_tuples(triples):
    return [(t.subject, t.predicate, t.object) for t in triples]


# -- abstraction / registry -------------------------------------------------

def test_registered_and_built_through_the_base_class():
    server = FakeServer()
    extractor = TripleExtractor('openie', server=server)

    assert isinstance(extractor, OpenIETripleExtractor)
    assert TripleExtractor.registry['openie'] is OpenIETripleExtractor
    assert extractor.strategy_name == 'openie'
    assert server.started


def test_close_and_context_manager_close_the_server():
    server = FakeServer()
    with TripleExtractor('openie', server=server):
        pass
    assert server.closed


def test_extract_rejects_empty_text():
    with pytest.raises(AssertionError):
        TripleExtractor('openie', server=FakeServer()).extract('   ')


def test_prompt_is_ignored():
    server = FakeServer(openie(('A', 'is', 'B1')))
    extractor = TripleExtractor('openie', server=server)
    assert as_tuples(extractor.extract('A is B1.', prompt='ignored')) == [
        ('A', 'is', 'B1')]


# -- request ----------------------------------------------------------------

def test_default_properties_sent():
    server = FakeServer()
    TripleExtractor('openie', server=server).extract('Some text.')

    _, props = server.requests[0]
    assert 'parse' in props['annotators'].split(',')
    assert 'openie' in props['annotators'].split(',')
    assert props['tokenize.options'] == 'splitHyphenated=false'


def test_properties_override_defaults():
    server = FakeServer()
    extractor = TripleExtractor(
        'openie', server=server, properties={'openie.triple.strict': 'false'})
    extractor.extract('Some text.')

    _, props = server.requests[0]
    assert props['openie.triple.strict'] == 'false'
    assert 'openie' in props['annotators']


@pytest.mark.parametrize('remove, expected', [
    (True, 'indentation, an extensive library.'),
    (False, 'indentation,[38] an extensive library.'),
])
def test_citation_removal(remove, expected):
    server = FakeServer()
    TripleExtractor('openie', server=server, remove_citations=remove).extract(
        'indentation,[38] an extensive library.')
    assert server.requests[0][0] == expected


@pytest.mark.parametrize('text, expected', [
    ('a[38] b', 'a b'),
    ('a[12, 13] b', 'a b'),
    ('a[3–5] b', 'a b'),
    ('a[citation needed] b', 'a b'),
    ('a[i] b', 'a[i] b'),
    ('a [b] c', 'a [b] c'),
])
def test_citation_regex(text, expected):
    assert CITATION_RE.sub('', text) == expected


# -- parentheticals ---------------------------------------------------------

@pytest.mark.parametrize('remove, expected', [
    (True, 'Alan Turing was an English mathematician.'),
    (False, 'Alan Turing (/ˈtjʊərɪŋ/; 23 June 1912 – 7 June 1954) was an English '
            'mathematician (and logician (sic)).'),
])
def test_parenthetical_removal(remove, expected):
    server = FakeServer()
    TripleExtractor('openie', server=server, remove_parentheticals=remove).extract(
        'Alan Turing (/ˈtjʊərɪŋ/; 23 June 1912 – 7 June 1954) was an English '
        'mathematician (and logician (sic)).')
    assert server.requests[0][0] == expected


# -- coreference ---------------------------------------------------------------

def mention(text, kind, sent, start, end, representative=False):
    """A CoreNLP coref mention (1-based sentence and token indices)."""
    return {'text': text, 'type': kind, 'sentNum': sent, 'startIndex': start,
            'endIndex': end, 'isRepresentativeMention': representative}


# "Alan Turing was a mathematician. He formalised algorithms and his work
#  helped the Allies. Bletchley Park hired him."
COREF_RESPONSE = {
    'sentences': [
        {'openie': [{'subject': 'Alan Turing', 'subjectSpan': [0, 2], 'relation': 'was',
                     'object': 'mathematician', 'objectSpan': [3, 4]}]},
        {'openie': [{'subject': 'He', 'subjectSpan': [0, 1], 'relation': 'formalised',
                     'object': 'algorithms', 'objectSpan': [2, 3]},
                    {'subject': 'his work', 'subjectSpan': [4, 6], 'relation': 'helped',
                     'object': 'Allies', 'objectSpan': [7, 8]}]},
        {'openie': [{'subject': 'Bletchley Park', 'subjectSpan': [0, 2], 'relation': 'hired',
                     'object': 'him', 'objectSpan': [2, 3]},
                    {'subject': 'It', 'subjectSpan': [0, 1], 'relation': 'is',
                     'object': 'unresolved', 'objectSpan': [2, 3]}]},
    ],
    'corefs': {
        '1': [mention('Alan Turing', 'PROPER', 1, 1, 3, representative=True),
              mention('He', 'PRONOMINAL', 2, 1, 2),
              mention('his', 'PRONOMINAL', 2, 5, 6),
              mention('him', 'PRONOMINAL', 3, 3, 4)],
        '2': [mention('It', 'PRONOMINAL', 3, 1, 2, representative=True),   # no antecedent
              mention('it', 'PRONOMINAL', 3, 9, 10)],
    },
}


def test_coref_replaces_pronouns_and_possessives():
    extractor = TripleExtractor('openie', server=FakeServer(COREF_RESPONSE))
    assert as_tuples(extractor.extract('...')) == [
        ('Alan Turing', 'was', 'mathematician'),
        ('Alan Turing', 'formalised', 'algorithms'),
        ("Alan Turing's work", 'helped', 'Allies'),
        ('Bletchley Park', 'hired', 'Alan Turing'),      # object pronoun
        ('It', 'is', 'unresolved'),                       # chain without a name
    ]


def test_coref_can_be_disabled():
    server = FakeServer(COREF_RESPONSE)
    extractor = TripleExtractor('openie', server=server, resolve_coref=False)
    subjects = [t.subject for t in extractor.extract('...')]

    assert 'He' in subjects and 'his work' in subjects
    annotators = server.requests[0][1]['annotators'].split(',')
    assert 'coref' not in annotators and 'ner' not in annotators


def test_coref_annotators_are_requested():
    server = FakeServer()
    TripleExtractor('openie', server=server).extract('Some text.')
    annotators = server.requests[0][1]['annotators'].split(',')
    assert annotators.index('ner') < annotators.index('coref') < annotators.index('openie')
    assert 'parse' in annotators
    assert server.requests[0][1]['coref.algorithm'] == 'statistical'


def test_coref_without_spans_keeps_text():
    response = {'sentences': [{'openie': [
        {'subject': 'He', 'relation': 'was', 'object': 'there'}]}],
        'corefs': COREF_RESPONSE['corefs']}
    extractor = TripleExtractor('openie', server=FakeServer(response))
    assert as_tuples(extractor.extract('...')) == [('He', 'was', 'there')]


# -- response parsing -------------------------------------------------------

def test_parse_dedups_normalizes_and_filters():
    server = FakeServer({'sentences': [
        {'openie': [
            {'subject': 'Barack  Obama', 'relation': 'was born in', 'object': 'Hawaii'},
            {'subject': 'Barack Obama', 'relation': 'was born in', 'object': ' Hawaii '},
            {'subject': 'X', 'relation': 'is', 'object': 'a'},     # too short
            {'subject': '', 'relation': 'is', 'object': 'empty'},  # no subject
        ]},
        {},                                                        # no openie key
        {'openie': [{'subject': 'He', 'relation': 'was', 'object': 'president'}]},
    ]})
    extractor = TripleExtractor('openie', server=server)

    assert extractor.extract('...') == [
        Triple('Barack Obama', 'was born in', 'Hawaii'),
        Triple('He', 'was', 'president'),
    ]


def test_min_object_chars():
    server = FakeServer(openie(('X', 'is', 'a')))
    extractor = TripleExtractor('openie', server=server, min_object_chars=0)
    assert as_tuples(extractor.extract('...')) == [('X', 'is', 'a')]

    with pytest.raises(ValueError):
        TripleExtractor('openie', server=FakeServer(), min_object_chars=-1)


VARIANTS = openie(
    ('Python', 'ease-of-writing with', 'use of significant indentation'),
    ('Python', 'ease-of-writing with', 'use of indentation'),
    ('Python', 'ease-of-writing with', 'use'),
    ('Python', 'supports', 'programming paradigms'),
    ('Python', 'supports', 'multiple programming paradigms'),
    ('Python', 'is', 'high-level'),
    ('Barack Obama', 'was', 'born'),
    ('Barack Obama', 'was born in', 'Hawaii'),
)


def test_remove_subsumed_keeps_the_most_specific_objects():
    extractor = TripleExtractor('openie', server=FakeServer(VARIANTS))
    assert as_tuples(extractor.extract('...')) == [
        ('Python', 'ease-of-writing with', 'use of significant indentation'),
        ('Python', 'supports', 'multiple programming paradigms'),
        ('Python', 'is', 'high-level'),
        ('Barack Obama', 'was', 'born'),             # different predicate
        ('Barack Obama', 'was born in', 'Hawaii'),
    ]


def test_remove_subsumed_can_be_disabled():
    extractor = TripleExtractor(
        'openie', server=FakeServer(VARIANTS), remove_subsumed=False)
    assert len(extractor.extract('...')) == 8


def test_predicate_variants_keep_the_shortest_predicate():
    triples = [
        Triple('Turing', 'also contributed Turing test to', 'mathematical biology'),
        Triple('Turing', 'contributed to', 'mathematical biology'),
        Triple('Turing', 'contributed Turing test to', 'mathematical biology'),
        Triple('Turing', 'also contributed', 'Turing test'),
        Triple('Turing', 'contributed', 'Turing test'),
        Triple('Turing', 'formalised concepts with', 'Turing machine'),
        Triple('Turing', 'formalised with', 'Turing machine'),
    ]
    assert as_tuples(OpenIETripleExtractor._drop_subsumed(triples)) == [
        ('Turing', 'contributed to', 'mathematical biology'),
        ('Turing', 'contributed', 'Turing test'),
        ('Turing', 'formalised with', 'Turing machine'),
    ]


def test_predicate_and_object_variants_combined():
    triples = [
        Triple('He', 'contributed Turing test to', 'philosophy'),
        Triple('He', 'contributed Turing test to', 'philosophy of artificial intelligence'),
        Triple('He', 'contributed to', 'philosophy'),
        Triple('He', 'contributed to', 'philosophy of artificial intelligence'),
    ]
    assert as_tuples(OpenIETripleExtractor._drop_subsumed(triples)) == [
        ('He', 'contributed to', 'philosophy of artificial intelligence')]


@pytest.mark.parametrize('negated', ['is not', 'was never', "is n't", 'cannot be'])
def test_negated_variants_are_kept(negated):
    positive = negated.split()[0] if negated != 'cannot be' else 'be'
    triples = [Triple('A', positive, 'B'), Triple('A', negated, 'B')]
    assert OpenIETripleExtractor._drop_subsumed(triples) == triples


def test_different_subjects_or_objects_are_not_merged():
    triples = [Triple('A', 'was born in', 'Hawaii'), Triple('A', 'was', 'born'),
               Triple('B', 'was born in', 'Hawaii'), Triple('A', 'born in', 'Kenya')]
    assert OpenIETripleExtractor._drop_subsumed(triples) == triples


def test_subsumption_needs_words_in_order():
    triples = [Triple('A', 'r', 'cat dog'), Triple('A', 'r', 'dog big cat')]
    assert OpenIETripleExtractor._drop_subsumed(triples) == triples


# -- CoreNLPServer: installation ---------------------------------------------

def make_home(parent, version):
    home = parent / f'stanford-corenlp-{version}'
    home.mkdir(parents=True)
    (home / f'stanford-corenlp-{version}.jar').write_bytes(b'')
    return home


def make_zip(path, version):
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr(f'stanford-corenlp-{version}/stanford-corenlp-{version}.jar', b'')
        z.writestr(f'stanford-corenlp-{version}/LICENSE.txt', 'x')
    return path


def test_find_installed_picks_newest_version(tmp_path):
    make_home(tmp_path, '4.5.3')
    newest = make_home(tmp_path, '4.10.0')    # numeric, not lexicographic
    (tmp_path / 'stanford-corenlp-9.9.9').mkdir()   # no jars: ignored
    assert corenlp.find_installed(tmp_path) == newest
    assert corenlp.find_installed(tmp_path / 'missing') is None


def test_resolve_home_from_folder(tmp_path, monkeypatch):
    monkeypatch.delenv('CORENLP_HOME', raising=False)
    home = make_home(tmp_path, '4.6.0')
    assert CoreNLPServer(home=home).resolve_home() == home


def test_resolve_home_from_zip_extracts_once(tmp_path, monkeypatch):
    monkeypatch.delenv('CORENLP_HOME', raising=False)
    zip_path = make_zip(tmp_path / 'corenlp.zip', '4.6.0')
    resources = tmp_path / 'res'
    server = CoreNLPServer(home=zip_path, resources=resources)

    home = server.resolve_home()
    assert home == resources / 'stanford-corenlp-4.6.0'
    assert corenlp.is_corenlp_home(home)

    (home / 'LICENSE.txt').write_text('kept')
    assert server.resolve_home() == home
    assert (home / 'LICENSE.txt').read_text() == 'kept'   # not re-extracted


def test_resolve_home_from_env(tmp_path, monkeypatch):
    home = make_home(tmp_path, '4.6.0')
    monkeypatch.setenv('CORENLP_HOME', str(home))
    assert CoreNLPServer(resources=tmp_path / 'empty').resolve_home() == home


def test_resolve_home_from_resources(tmp_path, monkeypatch):
    monkeypatch.delenv('CORENLP_HOME', raising=False)
    home = make_home(tmp_path, '4.6.0')
    assert CoreNLPServer(resources=tmp_path).resolve_home() == home


def test_resolve_home_errors(tmp_path, monkeypatch):
    monkeypatch.delenv('CORENLP_HOME', raising=False)
    with pytest.raises(FileNotFoundError):
        CoreNLPServer(home=tmp_path / 'nope').resolve_home()
    with pytest.raises(FileNotFoundError):
        CoreNLPServer(resources=tmp_path, auto_download=False).resolve_home()

    bad_zip = tmp_path / 'bad.zip'
    with zipfile.ZipFile(bad_zip, 'w') as z:
        z.writestr('a/x.txt', '')
        z.writestr('b/y.txt', '')
    with pytest.raises(RuntimeError):
        CoreNLPServer(home=bad_zip, resources=tmp_path / 'r').resolve_home()


def test_download_tries_mirrors_in_order(tmp_path, monkeypatch):
    calls = []
    source = make_zip(tmp_path / 'src.zip', '4.6.0')

    def fake_urlopen(url, timeout):
        calls.append(url)
        if url == 'https://dead':
            raise OSError('connection refused')
        return open(source, 'rb')

    monkeypatch.setattr(corenlp, 'CORENLP_URLS', ('https://dead', 'https://ok'))
    monkeypatch.setattr(corenlp.urllib.request, 'urlopen', fake_urlopen)

    home = corenlp.download(tmp_path / 'res')
    assert calls == ['https://dead', 'https://ok']
    assert home == tmp_path / 'res' / 'stanford-corenlp-4.6.0'
    assert not list((tmp_path / 'res').glob('*.part'))


def test_download_reports_all_failures(tmp_path, monkeypatch):
    def fake_urlopen(url, timeout):
        raise OSError(f'down: {url}')

    monkeypatch.setattr(corenlp, 'CORENLP_URLS', ('https://a', 'https://b'))
    monkeypatch.setattr(corenlp.urllib.request, 'urlopen', fake_urlopen)
    with pytest.raises(RuntimeError, match='https://a.*https://b'):
        corenlp.download(tmp_path)


# -- CoreNLPServer: lifecycle -------------------------------------------------

def test_start_reuses_running_server(monkeypatch):
    monkeypatch.setattr(CoreNLPServer, 'is_up', lambda self: True)
    server = CoreNLPServer().start()
    assert not server.owned
    server.close()   # must not try to stop a server it does not own


def test_busy_port_reuses_fallback_server_of_this_process(monkeypatch):
    """Port 9000 held by another program; a server was already started
    by this process on a fallback port (e.g. a notebook cell run again)."""
    monkeypatch.setattr(CoreNLPServer, '_fallback_ports',
                        {('localhost', 9000): 61234})
    monkeypatch.setattr(CoreNLPServer, 'is_up',
                        lambda self: self.port == 61234)
    monkeypatch.setattr(CoreNLPServer, '_port_in_use',
                        lambda self: self.port in (9000, 61234))
    monkeypatch.setattr(corenlp.shutil, 'which', lambda _: '/usr/bin/java')

    server = CoreNLPServer(port=9000).start()
    assert server.port == 61234
    assert not server.owned


def test_busy_port_is_detected():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        s.listen()
        port = s.getsockname()[1]
        assert CoreNLPServer(host='127.0.0.1', port=port)._port_in_use()
    assert not CoreNLPServer(host='127.0.0.1', port=port)._port_in_use()


# -- integration (needs Java + CoreNLP installed) -----------------------------

def _corenlp_available():
    if shutil.which('java') is None:
        return False
    try:
        CoreNLPServer(auto_download=False).resolve_home()
        return True
    except (FileNotFoundError, RuntimeError):
        return False


@pytest.fixture(scope='module')
def real_extractor():
    if not _corenlp_available():
        pytest.skip('Java and an installed CoreNLP are required')
    extractor = TripleExtractor('openie', auto_download=False)
    yield extractor
    extractor.close()


@pytest.mark.integration
def test_real_simple_sentence(real_extractor):
    triples = as_tuples(real_extractor.extract('Barack Obama was born in Hawaii.'))
    assert ('Barack Obama', 'was born in', 'Hawaii') in triples


@pytest.mark.integration
def test_real_long_sentence_with_citation(real_extractor):
    text = (
        'Python is a high-level, general-purpose programming language that '
        'emphasizes code readability, simplicity, and ease-of-writing with '
        'the use of significant indentation,[38] an extensive '
        '("batteries-included") standard library, and garbage collection. '
        'Python supports multiple programming paradigms but with an emphasis '
        'on object-oriented programming and dynamic typing')
    triples = as_tuples(real_extractor.extract(text))

    assert ('Python', 'supports', 'multiple programming paradigms') in triples
    assert all('38' not in part for t in triples for part in t)


@pytest.mark.integration
def test_real_pronouns_and_parentheticals(real_extractor):
    text = (
        'Alan Mathison Turing (/ˈtjʊərɪŋ/; 23 June 1912 – 7 June 1954) was an English '
        'mathematician and logician[6] widely regarded as the father of theoretical '
        'computer science.[7] He formalised the concepts of algorithm and computation '
        'with the Turing machine. He also contributed the Turing test to philosophy of '
        'artificial intelligence.')
    triples = as_tuples(real_extractor.extract(text))
    subjects = {s for s, _, _ in triples}

    assert ('Alan Mathison Turing', 'was regarded as',
            'father of theoretical computer science') in triples
    assert ('Alan Mathison Turing', 'contributed', 'Turing test') in triples
    assert 'He' not in subjects
