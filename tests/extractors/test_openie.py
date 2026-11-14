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
