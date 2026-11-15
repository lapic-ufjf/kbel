# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

"""Runs ReLiK in its own interpreter and talks to it over pipes."""

import atexit
import collections
import itertools
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)

WORKER = Path(__file__).with_name('worker.py')
DEFAULT_ENV = Path.home() / '.kbel' / 'relik-venv'
PYTHON_ENV_VAR = 'KBEL_RELIK_PYTHON'
SUPPORTED_PYTHONS = ('3.11', '3.10')


class ReLiKError(RuntimeError):
    pass


def env_python(env: Path) -> Path:
    return env / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def has_relik(python: str | Path) -> bool:
    try:
        return subprocess.run(
            [str(python), '-c', 'import relik'], capture_output=True, timeout=120).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def find_base_python() -> str | None:
    """A Python 3.10/3.11 interpreter to create the ReLiK environment with."""
    candidates = [shutil.which(f'python{v}') for v in SUPPORTED_PYTHONS]
    pyenv_root = Path(os.environ.get('PYENV_ROOT', Path.home() / '.pyenv')) / 'versions'
    if pyenv_root.is_dir():
        candidates += sorted((str(p / 'bin' / f'python{v}') for v in SUPPORTED_PYTHONS
                              for p in pyenv_root.glob(f'{v}*')), reverse=True)
    for python in filter(None, candidates):
        try:
            out = subprocess.run([python, '-c', 'import sys; print(sys.version_info[:2])'],
                                 capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if out.returncode == 0 and out.stdout.strip() in ('(3, 11)', '(3, 10)'):
            return python
    return None


def install(env: Path = DEFAULT_ENV, base_python: str | None = None,
            package: str = 'relik') -> Path:
    """Creates a virtual environment with ReLiK; returns its interpreter."""
    base_python = base_python or find_base_python()
    if base_python is None:
        raise ReLiKError(
            'ReLiK needs Python 3.10 or 3.11 (it pins torch 2.3), and none was found. '
            f'Install one (e.g. `pyenv install 3.11`), or point {PYTHON_ENV_VAR} to an '
            'interpreter where `pip install relik` was run.')
    LOG.warning('Installing ReLiK in %s (one time, ~2 GB) ...', env)
    env.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([base_python, '-m', 'venv', str(env)], check=True)
    python = env_python(env)
    subprocess.run([str(python), '-m', 'pip', 'install', '-q', '--upgrade', 'pip'], check=True)
    subprocess.run([str(python), '-m', 'pip', 'install', '-q', package], check=True)
    return python


def resolve_python(python: str | Path | None = None, auto_install: bool = True) -> Path:
    """The interpreter that runs ReLiK.

    Order: `python`, the `KBEL_RELIK_PYTHON` variable, `~/.kbel/relik-venv`,
    and finally (if `auto_install`) a new environment in that directory.
    """
    explicit = python or os.environ.get(PYTHON_ENV_VAR)
    if explicit:
        if not has_relik(explicit):
            raise ReLiKError(f'`import relik` fails with {explicit}.')
        return Path(explicit)
    default = env_python(DEFAULT_ENV)
    if default.exists() and has_relik(default):
        return default
    if not auto_install:
        raise ReLiKError(
            f'ReLiK is not installed. Set {PYTHON_ENV_VAR} to an interpreter with relik, '
            'or use auto_install=True.')
    return install()


class ReLiKWorker:
    """A ReLiK model loaded in a separate process.

    Requests are serialized (one at a time); the process is stopped by
    `close`, when the object is collected or when Python exits.
    """

    def __init__(
        self,
        model: str,
        python: str | Path | None = None,
        device: str = 'cpu',
        precision: str | None = None,
        model_options: dict[str, Any] | None = None,
        auto_install: bool = True,
        start_timeout: float = 1800,
    ):
        self.model = model
        self.python = resolve_python(python, auto_install)
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self._stderr: collections.deque[str] = collections.deque(maxlen=50)
        command = [str(self.python), str(WORKER), '--model', model, '--device', device,
                   '--options', json.dumps(model_options or {})]
        if precision:
            command += ['--precision', str(precision)]
        LOG.info('Starting ReLiK (%s) with %s', model, self.python)
        env = {**os.environ, 'PYTHONUNBUFFERED': '1', 'TOKENIZERS_PARALLELISM': 'false'}
        self._process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding='utf-8', bufsize=1, env=env)
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        atexit.register(self.close)
        self._await_ready(start_timeout)

    def _drain_stderr(self) -> None:
        for line in self._process.stderr:
            line = line.rstrip()
            if line:
                self._stderr.append(line)
                LOG.debug('[relik] %s', line)

    def _read(self, timeout: float | None) -> dict[str, Any]:
        result: dict[str, Any] = {}

        def read() -> None:
            result['line'] = self._process.stdout.readline()

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        reader.join(timeout)
        if reader.is_alive():
            self.close()
            raise ReLiKError(f'ReLiK did not answer in {timeout}s.')
        if not result.get('line'):
            raise ReLiKError('ReLiK stopped:\n' + '\n'.join(self._stderr))
        return json.loads(result['line'])

    def _await_ready(self, timeout: float) -> None:
        start = time.perf_counter()
        message = self._read(timeout)
        if not message.get('ready'):
            self.close()
            raise ReLiKError(f'Could not load {self.model}: {message.get("error")}\n'
                             f'{message.get("traceback", "")}')
        LOG.info('ReLiK ready in %.1fs', time.perf_counter() - start)

    @property
    def alive(self) -> bool:
        return self._process.poll() is None

    def annotate(self, texts: list[str], timeout: float | None = None,
                 **options: Any) -> list[dict[str, Any]]:
        """ReLiK outputs (spans and triplets) of each text, in order."""
        if not texts:
            return []
        with self._lock:
            if not self.alive:
                raise ReLiKError('ReLiK is not running:\n' + '\n'.join(self._stderr))
            request_id = next(self._ids)
            self._process.stdin.write(json.dumps(
                {'id': request_id, 'texts': texts, 'options': options}) + '\n')
            self._process.stdin.flush()
            message = self._read(timeout)
        if 'error' in message:
            raise ReLiKError(f'{message.get("type")}: {message["error"]}')
        if message.get('id') != request_id or len(message['outputs']) != len(texts):
            raise ReLiKError('Unexpected answer from the ReLiK worker.')
        return message['outputs']

    def close(self) -> None:
        process = getattr(self, '_process', None)
        if process is None or process.poll() is not None:
            return
        try:
            process.stdin.close()
            process.wait(timeout=10)
        except Exception:
            process.kill()

    def __del__(self) -> None:
        self.close()


if __name__ == '__main__':
    print(resolve_python(sys.argv[1] if len(sys.argv) > 1 else None))
