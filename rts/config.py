"""Configuration: paths, constants, and pinned choices for the RTS feasibility study.

See ``plan.md`` for the study design and ``implementation.md`` for tooling details.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Layout ---------------------------------------------------------------

WORKSPACE = Path(__file__).resolve().parent.parent
SUT = WORKSPACE / "sut" / "marshmallow"

MUTANTS_DIR = SUT / "mutants"
MUTATED_SRC = MUTANTS_DIR / "src"
ORIGINAL_SRC = SUT / "src"
TESTS_DIR = SUT / "tests"

STATS_FILE = MUTANTS_DIR / "mutmut-stats.json"
OUTCOMES_FILE = SUT / "mutmut-test-outcomes.jsonl"

# Full-suite relabelling (see plan_next_steps.md). mutmut runs only the tests covering the
# mutated function, so its log cannot see a fault whose killer lies outside that set. The
# full-suite log runs all 1190 collected tests per mutant and is written by
# ``scripts/emit_full_suite_outcomes.py``.
FULL_SUITE_OUTCOMES_FILE = SUT / "mutmut-full-suite-outcomes.jsonl"
FULL_SUITE_TESTS_FILE = SUT / "mutmut-full-suite-tests.json"

ARTIFACTS = WORKSPACE / "artifacts"

# --- Label source ---------------------------------------------------------

# Which outcome log defines the fault labels.
#   "mutmut" -- mutmut's selected tests only (the historical, documented default)
#   "full"   -- all 1190 collected tests (the corrected labels)
# Kept as module state rather than a parameter because it is a study-wide choice: every
# selector, feature and evaluation must agree on it, and threading it through ~20 call sites
# of ``dataset.build`` would be error-prone. Set with ``--labels`` on the CLIs, or the
# ``RTS_LABELS`` environment variable.
LABELS = os.environ.get("RTS_LABELS", "mutmut")


def set_labels(name: str) -> str:
    """Set the study-wide label source. Returns the previous value."""
    global LABELS
    if name not in ("mutmut", "full"):
        raise ValueError(f"unknown label source: {name!r}")
    previous, LABELS = LABELS, name
    return previous


def outcomes_file() -> Path:
    return FULL_SUITE_OUTCOMES_FILE if LABELS == "full" else OUTCOMES_FILE

# --- Pinned choices -------------------------------------------------------

# Fixed seed for the synthetic change ordering and for every stochastic model.
# The mutant population has no intrinsic temporal order, so history is imposed
# rather than observed. See implementation.md.
SEED = 20260924

# mutmut exit codes that mean "a test failed", i.e. the mutant was killed.
# 1 = pytest reported failures, 3 = pytest internal error.
KILLED_EXIT_CODES = {1, 3}
SURVIVED_EXIT_CODE = 0

# mutmut's own non-mutant runs, mirrored into MUTANT_UNDER_TEST.
NON_MUTANT_SENTINELS = {"", "mutant_generation", "fail", "stats"}

# --- Evaluation defaults --------------------------------------------------

DEFAULT_TRAIN_FRACTION = 0.8
DEFAULT_BUDGETS = (0.01, 0.02, 0.05, 0.1, 0.15, 0.2)
DEFAULT_BOOTSTRAP = 1000

# Candidate-set size per change. RTS ranks the whole suite, so this is the full
# number of collected tests.
FULL_SUITE = None

# --- SemIf ----------------------------------------------------------------

SEMIF_REPO = "https://github.com/TheoLeeCJ/SemIf-OpenJev"
SEMIF_MODEL = "Qwen/Qwen3-Reranker-4B"
SEMIF_MODEL_REVISION = "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
SEMIF_MAX_TOKENS = 8192
SEMIF_SCORES_FILE = ARTIFACTS / "semif_scores.jsonl"

# P1: SemIf's ``--mode direct`` checkpoint, pinned by the repo manifest at
# manifests/models.json (role ``direct_option_logits``).
DIRECT_MODEL = "Qwen/Qwen3.5-4B"
DIRECT_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

# P3: code-specialised encoder for the embedding baseline.
EMBED_MODEL = "microsoft/codebert-base"
EMBED_MODEL_REVISION = "3b0952feddeffad0063f274080e3c23d75e7eb39"


def ensure_artifacts_dir() -> Path:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS
