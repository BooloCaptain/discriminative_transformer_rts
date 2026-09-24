"""Configuration: paths, constants, and pinned choices for the RTS feasibility study.

See ``plan.md`` for the study design and ``implementation.md`` for tooling details.
"""

from __future__ import annotations

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

ARTIFACTS = WORKSPACE / "artifacts"

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


def ensure_artifacts_dir() -> Path:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS
