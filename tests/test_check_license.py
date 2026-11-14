# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import io
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_license.py"

_spec = importlib.util.spec_from_file_location("check_license", SCRIPT)
cl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cl)

HEADER = cl.HEADER
BODY = "import os\n\nprint(os.name)\n"


# -- header_status ------------------------------------------------------------

@pytest.mark.parametrize("text", [
    HEADER,
    HEADER + "\n" + BODY,
    "#!/usr/bin/env python3\n" + HEADER + BODY,
    "# -*- coding: utf-8 -*-\n" + HEADER + BODY,
    "#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n" + HEADER + BODY,
    HEADER.replace("\n", "\r\n") + BODY,          # Windows line endings
    "# Copyright 2026 LApIC   \n# SPDX-License-Identifier: Apache-2.0\n",
])
def test_status_ok(text):
    assert cl.header_status(text) == cl.OK


@pytest.mark.parametrize("text", [
    "",
    BODY,
    '"""Docstring."""\n' + BODY,
])
def test_status_missing(text):
    assert cl.header_status(text) == cl.MISSING


@pytest.mark.parametrize("text", [
    "# Copyright (c) 2026, lapic-ufjf\n# Licensed under MIT\n" + BODY,
    "# Copyright 2025 LApIC\n# SPDX-License-Identifier: Apache-2.0\n" + BODY,
    "# Copyright 2026 LApIC\n# SPDX-License-Identifier: MIT\n" + BODY,
    "# Copyright 2026 LApIC\n" + BODY,            # only half of it
    # Misplaced header: adding another one would duplicate it.
    "\n" + HEADER + BODY,                         # not on the first line
    BODY + HEADER,                                # right after the code
])
def test_status_invalid(text):
    assert cl.header_status(text) == cl.INVALID


# -- add_header ---------------------------------------------------------------

def test_add_header_plain_file():
    assert cl.add_header(BODY) == HEADER + "\n" + BODY


def test_add_header_empty_file():
    assert cl.add_header("") == HEADER


def test_add_header_strips_leading_blank_lines():
    assert cl.add_header("\n\n" + BODY) == HEADER + "\n" + BODY


def test_add_header_keeps_shebang_and_encoding_first():
    prolog = "#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n"
    assert cl.add_header(prolog + BODY) == prolog + HEADER + "\n" + BODY


def test_add_header_shebang_without_newline():
    assert cl.add_header("#!/usr/bin/env python3") == (
        "#!/usr/bin/env python3\n" + HEADER)


def test_add_header_is_idempotent():
    once = cl.add_header(BODY)
    assert cl.add_header(once) == once
    assert cl.header_status(once) == cl.OK


# -- iter_files ---------------------------------------------------------------

def test_iter_files_skips_excluded_and_non_python(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("")
    (tmp_path / "pkg" / "notes.txt").write_text("")
    for excluded in ("venv", "__pycache__", "kbel.egg-info"):
        (tmp_path / excluded).mkdir()
        (tmp_path / excluded / "x.py").write_text("")

    assert list(cl.iter_files([tmp_path])) == [tmp_path / "pkg" / "a.py"]


def test_iter_files_accepts_files_and_ignores_missing_paths(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("")
    assert list(cl.iter_files([f, tmp_path / "nope"])) == [f]


# -- check --------------------------------------------------------------------

@pytest.fixture
def project(tmp_path):
    (tmp_path / "good.py").write_text(HEADER + "\n" + BODY)
    (tmp_path / "missing.py").write_text(BODY)
    (tmp_path / "foreign.py").write_text(
        "# Copyright (c) 2026, someone\n" + BODY)
    return tmp_path


def test_check_reports_and_does_not_modify(project):
    before = {p.name: p.read_text() for p in project.iterdir()}
    out = io.StringIO()

    assert cl.check([project], out=out) == 1

    report = out.getvalue()
    assert f"Missing header: {project / 'missing.py'}" in report
    assert f"Invalid header: {project / 'foreign.py'}" in report
    assert "good.py" not in report
    assert "Files checked: 3" in report
    assert {p.name: p.read_text() for p in project.iterdir()} == before


def test_check_fix_adds_missing_but_leaves_foreign(project):
    foreign = (project / "foreign.py").read_text()

    assert cl.check([project], fix=True, out=io.StringIO()) == 1

    assert (project / "missing.py").read_text() == HEADER + "\n" + BODY
    assert (project / "foreign.py").read_text() == foreign


def test_check_passes_when_all_ok(project):
    (project / "foreign.py").unlink()
    cl.check([project], fix=True, out=io.StringIO())

    out = io.StringIO()
    assert cl.check([project], out=out) == 0
    assert "All files contain the expected license header." in out.getvalue()


# -- command line -------------------------------------------------------------

def test_cli_exit_codes(tmp_path):
    f = tmp_path / "a.py"
    f.write_text(BODY)

    def run(*args):
        return subprocess.run([sys.executable, str(SCRIPT), *args, str(f)],
                              capture_output=True, text=True).returncode

    assert run() == 1
    assert run("--fix") == 0
    assert run() == 0
    assert cl.header_status(f.read_text()) == cl.OK


def test_repository_files_have_header():
    """The project itself must comply (run `make license-add` to fix)."""
    out = io.StringIO()
    status = cl.check([ROOT / p for p in cl.DEFAULT_PATHS], out=out)
    assert status == 0, out.getvalue()
