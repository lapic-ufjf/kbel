# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

import atexit
import json
import logging
import os
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Optional

from kbel.core.triple import Triple

from ..abc import TripleExtractor

LOG = logging.getLogger(__name__)

CORENLP_VERSION = '4.5.3'
CORENLP_URL = f'https://nlp.stanford.edu/software/stanford-corenlp-{CORENLP_VERSION}.zip'
DEFAULT_RESOURCES = Path('~/.stanfordnlp_resources').expanduser()

DEFAULT_PROPERTIES: dict[str, Any] = {
    'annotators': 'tokenize,ssplit,pos,lemma,depparse,natlog,openie',
    'outputFormat': 'json',
    'openie.affinity_probability_cap': 2 / 3,
}


class OpenIETripleExtractor(TripleExtractor, strategy_name='openie'):
    """Triple extractor based on Stanford OpenIE (CoreNLP server).

    Rule-based and deterministic: no LLM calls. English only. Requires Java.

    The server is handled automatically:
      1. If a CoreNLP server is already answering on `host:port`, it is reused
         (and never killed by this process).
      2. Otherwise CoreNLP is located (`CORENLP_HOME`, or the default
         `~/.stanfordnlp_resources/stanford-corenlp-4.5.3`, downloaded if
         missing) and a server is started in the background.
      3. A server started here is stopped automatically at interpreter exit.

    Example:
        >>> extractor = TripleExtractor('openie')
        >>> extractor.extract("Barack Obama was born in Hawaii.")
        [Triple(subject='Barack Obama', predicate='was born in', object='Hawaii')]
    """

    def __init__(
        self,
        strategy_name: str,
        host: str = 'localhost',
        port: int = 9000,
        memory: str = '4G',
        corenlp_home: Optional[str | Path] = None,
        auto_download: bool = True,
        properties: Optional[dict[str, Any]] = None,
        start_timeout: float = 180,
        request_timeout: float = 120,
        min_object_chars: int = 2,
        graph: bool = False,
        graph_path: str | Path = 'graph.png',
    ):
        """Initializes the OpenIETripleExtractor.

        Args:
            strategy_name (str): Name of this strategy.
            host (str): CoreNLP server host.
            port (int): CoreNLP server port.
            memory (str): Max JVM memory (`-Xmx`) if a server must be started.
            corenlp_home (str | Path): CoreNLP folder. Defaults to the
                `CORENLP_HOME` env var, then to the default resources folder.
            auto_download (bool): Download CoreNLP if it is not found.
            properties (dict): CoreNLP properties, merged over the defaults.
            start_timeout (float): Seconds to wait for the server to be ready.
            request_timeout (float): Seconds to wait for each request.
            min_object_chars (int): Triples whose object is shorter than this
                are discarded.
            graph (bool): Render a graph image on every `extract` call.
            graph_path (str | Path): Where to save the graph image.
        """
        assert strategy_name == self.strategy_name
        super().__init__(graph=graph, graph_path=graph_path)

        self._url = f'http://{host}:{port}'
        self._port = port
        self._memory = memory
        self._home = corenlp_home
        self._auto_download = auto_download
        self._properties = {**DEFAULT_PROPERTIES, **(properties or {})}
        self._start_timeout = start_timeout
        self._request_timeout = request_timeout
        self._min_object_chars = min_object_chars
        self._proc: Optional[subprocess.Popen] = None

        self._ensure_server()

    # -- server management -------------------------------------------------

    def _is_up(self) -> bool:
        try:
            with urllib.request.urlopen(f'{self._url}/ping', timeout=2) as r:
                return r.status == 200
        except Exception:
            return False

    def _resolve_home(self) -> Path:
        home = Path(
            self._home
            or os.environ.get('CORENLP_HOME')
            or DEFAULT_RESOURCES / f'stanford-corenlp-{CORENLP_VERSION}'
        ).expanduser()
        if home.is_dir():
            return home
        if not self._auto_download:
            raise FileNotFoundError(
                f'CoreNLP not found at {home}. Set CORENLP_HOME or enable auto_download.')

        LOG.info(f'Downloading CoreNLP {CORENLP_VERSION} to {home.parent} (one-time, large file)...')
        home.parent.mkdir(parents=True, exist_ok=True)
        zip_path = home.parent / f'stanford-corenlp-{CORENLP_VERSION}.zip'
        try:
            urllib.request.urlretrieve(CORENLP_URL, zip_path)
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(home.parent)
        except Exception as e:
            raise RuntimeError(
                f'Could not download CoreNLP from {CORENLP_URL}: {e}. '
                f'Download it manually and set CORENLP_HOME.') from e
        finally:
            zip_path.unlink(missing_ok=True)
        return home

    def _ensure_server(self) -> None:
        if self._is_up():
            LOG.info(f'Reusing CoreNLP server at {self._url}')
            return

        if shutil.which('java') is None:
            raise RuntimeError('Java (JRE) not found in PATH. Install it to use the openie extractor.')

        home = self._resolve_home()
        cmd = [
            'java', f'-Xmx{self._memory}',
            '-cp', str(home / '*'),
            'edu.stanford.nlp.pipeline.StanfordCoreNLPServer',
            '-port', str(self._port),
            '-timeout', '60000',
            '-quiet', 'true',
        ]
        LOG.info(f'Starting CoreNLP server at {self._url}')
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        atexit.register(self.close)

        deadline = time.monotonic() + self._start_timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                self._proc = None
                raise RuntimeError(
                    'CoreNLP server exited during startup (port in use or not enough memory?).')
            if self._is_up():
                return
            time.sleep(0.5)

        self.close()
        raise TimeoutError(f'CoreNLP server not ready after {self._start_timeout}s.')

    def close(self) -> None:
        """Stops the server, only if it was started by this extractor."""
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    def __enter__(self) -> 'OpenIETripleExtractor':
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- extraction --------------------------------------------------------

    def _extract(self, text: str, prompt=None) -> list[Triple]:
        """Extracts triples from a text using Stanford OpenIE.

        Args:
            text (str): Input text.
            prompt: Ignored (kept for interface compatibility).

        Returns:
            list[Triple]: Deduplicated list of extracted triples.
        """
        if prompt is not None:
            LOG.debug('`prompt` has no effect in the openie extractor.')

        url = f'{self._url}/?' + urllib.parse.urlencode(
            {'properties': json.dumps(self._properties)})
        req = urllib.request.Request(url, data=text.encode('utf-8'), method='POST')
        try:
            with urllib.request.urlopen(req, timeout=self._request_timeout) as r:
                data = json.loads(r.read().decode('utf-8'))
        except Exception as e:
            LOG.warning(f'Exception occurred while extracting triples: {e}')
            raise

        triples: dict[tuple[str, str, str], Triple] = {}
        for sentence in data.get('sentences', []):
            for t in sentence.get('openie', []):
                s = t.get('subject', '').strip()
                p = t.get('relation', '').strip()
                o = t.get('object', '').strip()
                if not (s and p and o) or len(o) < self._min_object_chars:
                    continue
                triples.setdefault((s, p, o), Triple(subject=s, predicate=p, object=o))
        return list(triples.values())