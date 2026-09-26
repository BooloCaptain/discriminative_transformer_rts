"""Root conftest: makes the workspace root importable when running pytest.

Tests live in ``tests/`` without an ``__init__.py``, so pytest would otherwise insert
only ``tests/`` on ``sys.path`` and ``import rts`` would fail.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
