"""Study configuration: pinned checkpoints, the SUT checkout, and artifact paths.

These are *this study's* choices. They live here rather than in :mod:`rts.config` so the
harness kernel carries no study pin, and so two studies can disagree about a checkpoint,
a checkout or a label source without editing the kernel.
"""

from __future__ import annotations

# Re-exported from the kernel so a study module can read one config object. The explicit
# ``X as X`` form marks them as re-exports rather than unused imports.
from rts.config import (
    ARTIFACTS as ARTIFACTS,
)
from rts.config import (
    DEFAULT_TRAIN_FRACTION as DEFAULT_TRAIN_FRACTION,
)
from rts.config import (
    SEED as SEED,
)
from rts.config import (
    WORKSPACE as WORKSPACE,
)
from rts.config import (
    ensure_artifacts_dir as ensure_artifacts_dir,
)

# --- Layout ---------------------------------------------------------------

SUT = WORKSPACE / "sut" / "marshmallow"

MUTANTS_DIR = SUT / "mutants"
MUTATED_SRC = MUTANTS_DIR / "src"
ORIGINAL_SRC = SUT / "src"
TESTS_DIR = SUT / "tests"

STATS_FILE = MUTANTS_DIR / "mutmut-stats.json"
OUTCOMES_FILE = SUT / "mutmut-test-outcomes.jsonl"

# Full-suite relabelling: mutmut runs only the tests covering the mutated function, so its
# log cannot see a fault whose killer lies outside that set. The full-suite log runs all
# collected tests per mutant.
FULL_SUITE_OUTCOMES_FILE = SUT / "mutmut-full-suite-outcomes.jsonl"
FULL_SUITE_TESTS_FILE = SUT / "mutmut-full-suite-tests.json"

# Valid label sources for the mutation dataset.
LABEL_SOURCES = ("mutmut", "full")

# mutmut exit codes that mean "a test failed", i.e. the mutant was killed.
KILLED_EXIT_CODES = {1, 3}
SURVIVED_EXIT_CODE = 0

# mutmut's own non-mutant runs, mirrored into MUTANT_UNDER_TEST.
NON_MUTANT_SENTINELS = {"", "mutant_generation", "fail", "stats"}

# --- SemIf ----------------------------------------------------------------

SEMIF_REPO = "https://github.com/TheoLeeCJ/SemIf-OpenJev"
SEMIF_MODEL = "Qwen/Qwen3-Reranker-4B"
SEMIF_MODEL_REVISION = "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
SEMIF_MAX_TOKENS = 8192
SEMIF_SCORES_FILE = ARTIFACTS / "semif_scores.jsonl"

# P1: SemIf's ``--mode direct`` checkpoint, pinned by the repo manifest.
DIRECT_MODEL = "Qwen/Qwen3.5-4B"
DIRECT_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

# P3: code-specialised encoder for the embedding baseline.
EMBED_MODEL = "microsoft/codebert-base"
EMBED_MODEL_REVISION = "3b0952feddeffad0063f274080e3c23d75e7eb39"
