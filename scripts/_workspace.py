"""Locate the workspace root, so a script need not count its own parents.

A script that computes ``Path(__file__).resolve().parent.parent`` breaks the moment it is moved
one directory deeper, and it breaks *quietly*: the constant becomes a path to the wrong
directory, so the failure surfaces later as a missing dataset rather than as an import error.
Searching upward for a marker makes where the script lives irrelevant.

The marker pair is deliberate: ``rts/`` for the package and ``conftest.py`` for the root, so a
copy of ``rts/`` inside a checkout under ``sut/`` cannot be mistaken for the workspace.
"""

from __future__ import annotations

from pathlib import Path


def find_workspace(start: Path | None = None) -> Path:
    here = (start or Path(__file__).resolve().parent).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "rts" / "__init__.py").is_file() and (candidate / "conftest.py").is_file():
            return candidate
    raise RuntimeError(f"could not find the workspace root above {here}")


WORKSPACE = find_workspace()
