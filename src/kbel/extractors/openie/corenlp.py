# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

"""Minimal client for a Stanford CoreNLP server (standard library only).

`CoreNLPServer` finds (or downloads) a CoreNLP installation, starts the
Java server when needed and sends annotation requests to it. It knows
nothing about triples: that is the job of `OpenIETripleExtractor`.
"""

import atexit
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Optional

LOG = logging.getLogger(__name__)

CORENLP_VERSION = '4.6.0'
CORENLP_URLS: tuple[str, ...] = (
    f'https://downloads.cs.stanford.edu/nlp/software/stanford-corenlp-{CORENLP_VERSION}.zip',
    'https://huggingface.co/stanfordnlp/CoreNLP/resolve/main/stanford-corenlp-latest.zip',
)
DEFAULT_RESOURCES = Path('~/.stanfordnlp_resources').expanduser()
DOWNLOAD_TIMEOUT = 60  # seconds without data before a mirror is skipped

_HOME_DIR_RE = re.compile(r'^stanford-corenlp-(\d+(?:\.\d+)*)$')


class CoreNLPServer:
    """A CoreNLP server: reused if already running, otherwise started here.

    Resolution of the CoreNLP installation (only needed to start a server):
      1. `home` (or the `CORENLP_HOME` env var): the CoreNLP folder, or its
         `.zip`, which is extracted into `resources` on first use;
      2. the newest `stanford-corenlp-*` folder inside `resources`;
      3. a download into `resources` (if `auto_download`).

    A server started here is stopped by `close()`, on context-manager exit
    or at interpreter exit. A server that was already running is never
    stopped. If the port is held by another program, a free port is used;
    it is remembered for the process, so building another server for the
    same host/port reuses it instead of starting a new JVM.

    Parameters:
        host: Server host.
        port: Server port.
        home: CoreNLP folder or zip.
        resources: Where CoreNLP is looked up / extracted / downloaded.
        auto_download: Download CoreNLP if no installation is found.
        memory: Max JVM memory (`-Xmx`).
        start_timeout: Seconds to wait for a started server to answer.
        request_timeout: Seconds to wait for each annotation request.
    """

    #: (host, requested port) -> port actually used, when it was taken.
    _fallback_ports: dict[tuple[str, int], int] = {}

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 9000,
        home: Optional[str | Path] = None,
        resources: str | Path = DEFAULT_RESOURCES,
        auto_download: bool = True,
        memory: str = '4G',
        start_timeout: float = 180,
        request_timeout: float = 120,
    ):
        self.host = host
        self.port = port
        self._home = home
        self._resources = Path(resources).expanduser()
        self._auto_download = auto_download
        self._memory = memory
        self._start_timeout = start_timeout
        self._request_timeout = request_timeout
        self._proc: Optional[subprocess.Popen] = None

    @property
    def url(self) -> str:
        return f'http://{self.host}:{self.port}'

    @property
    def owned(self) -> bool:
        """Whether the server process was started (and is owned) by us."""
        return self._proc is not None

    # -- lifecycle ---------------------------------------------------------

    def is_up(self) -> bool:
        """Whether a CoreNLP server answers on `url`."""
        try:
            with urllib.request.urlopen(f'{self.url}/ping', timeout=2) as r:
                return r.status == 200
        except Exception:
            return False

    def start(self) -> 'CoreNLPServer':
        """Ensure a server is answering; start one if needed."""
        if self.is_up():
            LOG.info(f'Reusing CoreNLP server at {self.url}')
            return self
        if shutil.which('java') is None:
            raise RuntimeError(
                'Java (JRE) not found in PATH; it is required by CoreNLP.')

        requested = (self.host, self.port)
        if self._port_in_use():
            busy = self.port
            previous = CoreNLPServer._fallback_ports.get(requested)
            if previous is not None:
                self.port = previous
                if self.is_up():
                    LOG.warning(f'Port {busy} is in use by another program; '
                                f'reusing CoreNLP server at {self.url}')
                    return self
            self.port = _free_port()
            LOG.warning(f'Port {busy} is in use by another program; '
                        f'starting CoreNLP on port {self.port} instead.')

        home = self.resolve_home()

        cmd = [
            'java', f'-Xmx{self._memory}', '-cp', str(home / '*'),
            'edu.stanford.nlp.pipeline.StanfordCoreNLPServer',
            '-port', str(self.port), '-timeout', '60000', '-quiet', 'true',
        ]
        LOG.info(f'Starting CoreNLP server ({home.name}) at {self.url}')
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        atexit.register(self.close)

        deadline = time.monotonic() + self._start_timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                self._proc = None
                raise RuntimeError(
                    'CoreNLP server exited during startup '
                    '(not enough memory, or port taken?).')
            if self.is_up():
                if requested != (self.host, self.port):
                    CoreNLPServer._fallback_ports[requested] = self.port
                return self
            time.sleep(0.5)

        self.close()
        raise TimeoutError(
            f'CoreNLP server not ready after {self._start_timeout}s.')

    def close(self) -> None:
        """Stop the server, only if it was started by us."""
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    def __enter__(self) -> 'CoreNLPServer':
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()


    def annotate(self, text: str, properties: dict[str, Any]) -> dict:
        """Annotate `text`; returns CoreNLP's JSON output."""
        props = {**properties, 'outputFormat': 'json'}
        url = f'{self.url}/?' + urllib.parse.urlencode(
            {'properties': json.dumps(props)})
        req = urllib.request.Request(
            url, data=text.encode('utf-8'), method='POST')
        with urllib.request.urlopen(req, timeout=self._request_timeout) as r:
            return json.loads(r.read().decode('utf-8'))


    def resolve_home(self) -> Path:
        """Locate (extracting or downloading if needed) CoreNLP."""
        configured = self._home or os.environ.get('CORENLP_HOME')
        if configured:
            path = Path(configured).expanduser()
            if path.is_file() and path.suffix == '.zip':
                return extract_zip(path, self._resources)
            if is_corenlp_home(path):
                return path
            raise FileNotFoundError(
                f'No CoreNLP found at {path} (expected the CoreNLP folder '
                'or its .zip).')

        installed = find_installed(self._resources)
        if installed is not None:
            return installed
        if not self._auto_download:
            raise FileNotFoundError(
                f'CoreNLP not found in {self._resources}. Pass `home` '
                '(folder or .zip), set CORENLP_HOME or enable auto_download.')
        return download(self._resources)

    def _port_in_use(self) -> bool:
        try:
            with socket.create_connection((self.host, self.port), timeout=1):
                return True
        except OSError:
            return False


def is_corenlp_home(path: Path) -> bool:
    """Whether `path` is a CoreNLP folder (contains its jars)."""
    return path.is_dir() and any(path.glob('stanford-corenlp-*.jar'))


def find_installed(resources: Path) -> Optional[Path]:
    """Newest `stanford-corenlp-X.Y.Z` folder in `resources`, if any."""
    found = []
    if resources.is_dir():
        for d in resources.iterdir():
            m = _HOME_DIR_RE.match(d.name)
            if m and is_corenlp_home(d):
                found.append((tuple(map(int, m[1].split('.'))), d))
    return max(found)[1] if found else None


def extract_zip(zip_path: Path, dest: Path) -> Path:
    """Extract a CoreNLP zip into `dest` (once); returns the CoreNLP folder."""
    with zipfile.ZipFile(zip_path) as z:
        top = {n.split('/', 1)[0] for n in z.namelist() if n.strip('/')}
        if len(top) != 1:
            raise RuntimeError(f'{zip_path} does not look like a CoreNLP zip.')
        home = dest / top.pop()
        if not is_corenlp_home(home):
            LOG.info(f'Extracting {zip_path} to {dest}...')
            dest.mkdir(parents=True, exist_ok=True)
            z.extractall(dest)
    if not is_corenlp_home(home):
        raise RuntimeError(f'No CoreNLP jars found in {zip_path}.')
    return home


def download(dest: Path) -> Path:
    """Download and extract CoreNLP into `dest`, trying each mirror."""
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / 'stanford-corenlp-download.zip.part'
    errors = []
    for url in CORENLP_URLS:
        LOG.info(f'Downloading CoreNLP from {url} (one-time, ~500 MB)...')
        try:
            with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as r, \
                    open(zip_path, 'wb') as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            return extract_zip(zip_path, dest)
        except Exception as e:
            LOG.warning(f'Download from {url} failed: {e}')
            errors.append(f'{url}: {e}')
        finally:
            zip_path.unlink(missing_ok=True)
    raise RuntimeError(
        'Could not download CoreNLP (' + '; '.join(errors) + '). Download '
        'the zip manually and pass it as `home` (or set CORENLP_HOME).')


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]
