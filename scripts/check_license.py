#!/usr/bin/env python3
# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

"""Check (and optionally add) the license header of the project's files.

Every Python file must start with::

    # Copyright 2026 LApIC
    # SPDX-License-Identifier: Apache-2.0

A shebang (``#!``) and/or an encoding declaration (``# -*- coding: ... -*-``)
may come before the header.

Usage:
    python scripts/check_license.py [--fix] [PATH ...]

Without ``--fix`` it only reports files without the header and exits with
status 1 if any is found. With ``--fix`` the header is added to the files
that are missing it. Files that already carry a *different* copyright or
SPDX line are reported as invalid and never modified automatically.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

HEADER_LINES: tuple[str, ...] = (
    "# Copyright 2026 LApIC",
    "# SPDX-License-Identifier: Apache-2.0",
)
HEADER = "\n".join(HEADER_LINES) + "\n"

DEFAULT_PATHS: tuple[str, ...] = ("src", "tests", "scripts")
EXTENSIONS: frozenset[str] = frozenset({".py"})
EXCLUDED_DIRS: frozenset[str] = frozenset({
    ".git", ".venv", "venv", "__pycache__", "build", "dist",
    ".pytest_cache", ".ruff_cache", ".mypy_cache",
})

_ENCODING_RE = re.compile(r"^#.*coding[:=]")
_FOREIGN_HEADER_RE = re.compile(r"^#.*(copyright|spdx-license-identifier)",
                                re.IGNORECASE)

# Header status.
OK = "ok"
MISSING = "missing"
INVALID = "invalid"


def iter_files(paths: Iterable[str | Path]) -> Iterator[Path]:
    """Yield the files to check under `paths`, sorted, skipping excluded dirs."""
    for root in map(Path, paths):
        if root.is_file():
            if root.suffix in EXTENSIONS:
                yield root
            continue
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            rel_parts = path.relative_to(root).parts
            if any(part in EXCLUDED_DIRS or part.endswith(".egg-info")
                   for part in rel_parts):
                continue
            if path.is_file() and path.suffix in EXTENSIONS:
                yield path


def _prolog_length(lines: list[str]) -> int:
    """Number of leading lines (shebang/encoding) allowed before the header."""
    n = 0
    if n < len(lines) and lines[n].startswith("#!"):
        n += 1
    if n < len(lines) and _ENCODING_RE.match(lines[n]):
        n += 1
    return n


def header_status(text: str) -> str:
    """Return OK, MISSING or INVALID for the given file content."""
    lines = text.splitlines()
    start = _prolog_length(lines)
    head = [line.rstrip() for line in lines[start:start + len(HEADER_LINES)]]
    if tuple(head) == HEADER_LINES:
        return OK
    # A different copyright/SPDX line near the top: don't touch it.
    if any(_FOREIGN_HEADER_RE.match(line) for line in lines[start:start + 5]):
        return INVALID
    return MISSING


def add_header(text: str) -> str:
    """Return `text` with the license header inserted after any prolog."""
    if header_status(text) == OK:
        return text
    lines = text.splitlines(keepends=True)
    start = _prolog_length([line.rstrip("\r\n") for line in lines])
    prolog = "".join(lines[:start])
    if prolog and not prolog.endswith("\n"):
        prolog += "\n"
    body = "".join(lines[start:])
    if not body.strip():
        return prolog + HEADER
    return prolog + HEADER + "\n" + body.lstrip("\n")


def check(paths: Iterable[str | Path], fix: bool = False,
          out=sys.stdout) -> int:
    """Check `paths`; returns the process exit status (0 = all good)."""
    total = fixed = 0
    problems: list[tuple[Path, str]] = []

    for path in iter_files(paths):
        total += 1
        text = path.read_text(encoding="utf-8")
        status = header_status(text)
        if status == OK:
            continue
        if status == MISSING and fix:
            path.write_text(add_header(text), encoding="utf-8")
            fixed += 1
            print(f"Added header: {path}", file=out)
            continue
        problems.append((path, status))

    for path, status in problems:
        label = "Missing header" if status == MISSING else "Invalid header"
        print(f"{label}: {path}", file=out)

    print(f"Files checked: {total}", file=out)
    if fix:
        print(f"Headers added: {fixed}", file=out)

    if problems:
        print(f"Files with issues: {len(problems)}", file=out)
        if any(status == MISSING for _, status in problems):
            print("Run 'make license-add' to add the missing headers.",
                  file=out)
        if any(status == INVALID for _, status in problems):
            print("Files with a different copyright/SPDX header must be "
                  "fixed by hand.", file=out)
        return 1

    print("All files contain the expected license header.", file=out)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", default=list(DEFAULT_PATHS),
                        help="files or directories to check "
                             f"(default: {' '.join(DEFAULT_PATHS)})")
    parser.add_argument("--fix", action="store_true",
                        help="add the header to files that are missing it")
    args = parser.parse_args(argv)
    return check(args.paths, fix=args.fix)


if __name__ == "__main__":
    sys.exit(main())
