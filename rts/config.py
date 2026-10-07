"""Configuration: paths and the evaluation defaults the harness shares.

A study's own pins -- checkpoints, a checkout, a label source -- do not belong here; they live
with that study (see ``examples/config.py``). What is here is what the *kernel* needs: where the
workspace is, where artifacts are written, and the default evaluation knobs.

The label source used to be a module global here, on the argument that every ranker, feature and
evaluation must agree on it. The argument was right and the mechanism was wrong: agreement is
achieved by handing one source object to every consumer of a run, so no global can leak between
two datasets, two label sources or two tests.
"""

from __future__ import annotations

from pathlib import Path

# --- Layout ---------------------------------------------------------------

WORKSPACE = Path(__file__).resolve().parent.parent
ARTIFACTS = WORKSPACE / "artifacts"

# --- Evaluation defaults --------------------------------------------------

#: The default seed for a run's stochastic choices: the split shuffle and every model that
#: takes a seed. A dataset's *own* order is a property of the data, not of this.
SEED = 20260924

DEFAULT_TRAIN_FRACTION = 0.8
DEFAULT_BUDGETS = (0.01, 0.02, 0.05, 0.1, 0.15, 0.2)
DEFAULT_BOOTSTRAP = 1000


def ensure_artifacts_dir() -> Path:
    """Create the artifacts directory if absent, and return it."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS
