"""The fast check: what to run before every commit, when the full gate is too slow.

The full reproduction gate (``scripts/verify_experiment_layer.py``) is the acceptance
criterion -- it re-runs every migrated condition and compares the artifact it renders leaf by leaf.
That takes about half an hour here, almost all of it fitting XGBoost trees and reading SemIf
caches, so it is not something to run per commit.

This runs the cheap *part* of that check instead, in tiers, and says what each tier does and
does not cover:

    tier 1  static      ruff on the declared config, and every module imports          few seconds
    tier 2  unit        the test suite                                                  ~35 s
    tier 3  numbers     the headline condition's *non-fitting* rankers, rendered through the
                        real driver and compared against ``results_full.json``         ~1-2 min
    tier 4  real data   the BugsInPy condition end to end (``--with-bugsinpy``)               ~2-3 min

Tier 3 is the one worth understanding. It runs the real experiment layer over the real
marshmallow dataset and the real artifact, with the same controls the headline condition recorded
(budgets, seed, bootstrap count, ``history`` override, ``full`` candidate mode), but with a
model axis of the nine rankers that need no fitting. Every leaf of those nine design points -- the
per-budget recall, its interval, the paired deltas against ``coverage`` -- is then compared
against the recorded artifact, and the contrast is made through ``render.pipeline``, so the
renderer is exercised too.

That is a strong check for its cost: it is the same code path as the full gate, and it
includes the ``random`` row, whose exact reproduction is the load-bearing evidence that the
RNG *consumption pattern* has not moved -- the failure mode ``docs/refactor.md`` §12 records.
It does **not** cover: the fitted models (XGBoost, SemIf), the low_cooccurrence and covered conditions, the
ladder, or the variation sections. Those are the full gate's job; run it before you trust a
change to a model, a feature, or a cache.

Usage::

    python scripts/check_fast.py                  # tiers 1-3
    python scripts/check_fast.py --quick          # tiers 1-2, no dataset needed
    python scripts/check_fast.py --with-bugsinpy  # tiers 1-4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _workspace import WORKSPACE  # noqa: E402

sys.path.insert(0, str(WORKSPACE))

from rts import config, studies  # noqa: E402
from rts.experiment import (  # noqa: E402
    FACTOR_MODEL,
    Contrast,
    Controls,
    Experiment,
    run,
)
from rts.model.rankers import (  # noqa: E402
    CoverageRanker,
    FailureRateRanker,
    LexicalRanker,
    RandomRanker,
    RecencyRanker,
    StructuralRuleRanker,
)
from rts.render import pipeline  # noqa: E402

#: The headline condition's rankers that need no fitting: rules, BM25, and the BM25 shuffle
#: controls. ``default_rankers`` minus the trees and the reranker.
CHEAP = (
    RandomRanker(),
    RecencyRanker(),
    FailureRateRanker(),
    CoverageRanker(),
    StructuralRuleRanker(),
    LexicalRanker(),
)


def _tier(number: int, title: str, covers: str) -> None:
    print(f"\n[tier {number}] {title}  --  {covers}", flush=True)


def tier_static() -> None:
    _tier(1, "static", "ruff on the declared config, and every module imports")
    ruff = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "rts/", "tests/", "scripts/", "conftest.py"],
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
    )
    if ruff.returncode != 0:
        print(ruff.stdout or ruff.stderr)
        raise SystemExit("tier 1: ruff is not clean")
    print("  ruff: clean")

    import importlib

    modules = sorted(
        "rts." + ".".join(p.relative_to(WORKSPACE / "rts").with_suffix("").parts)
        for p in (WORKSPACE / "rts").rglob("*.py")
        if p.name != "__init__.py" and "__pycache__" not in p.parts
    )
    for name in modules:
        importlib.import_module(name)
    print(f"  imports: {len(modules)} modules")


def tier_unit() -> None:
    _tier(2, "unit", "the test_suite, over the stub dataset")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-o", "addopts=", "-q"],
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
    )
    tail = (result.stdout or "").strip().splitlines()[-1:] or ["(no output)"]
    print(f"  {tail[0]}")
    if result.returncode != 0:
        print(result.stdout)
        raise SystemExit("tier 2: the test_suite is not green")


def cheap_condition() -> Experiment:
    """The headline condition restricted to the rankers that need no fitting.

    Declared here rather than in ``rts/studies`` because it is a *check*, not a study choice:
    the sweep is data in ``rts/studies``, and a caller may declare its own experiment. Every
    control matches what the recorded artifact was produced with, which is what makes the
    contrast meaningful.
    """
    controls = [
        LexicalRanker(**kwargs) for _, kwargs in studies.ABLATION_MODELS
    ]
    return Experiment(
        name="check.fast.study",
        datasets=studies.dataset_axis(("mutmut",)),
        features=studies.structured_feature_axis(),
        models=studies.model_axis(CHEAP + tuple(controls)),
        subsets=studies.subset_axis(("detectable",)),
        splits=studies.split_axis(),
        controls=Controls(
            seed=config.SEED,
            budgets=config.DEFAULT_BUDGETS,
            n_bootstrap=config.DEFAULT_BOOTSTRAP,
            candidate_sets="full",
        ),
        contrasts=(Contrast(FACTOR_MODEL, "coverage", 0.05),),
        temporal=True,
    )


def leaves(value, path: str = ""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from leaves(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from leaves(item, f"{path}[{index}]")
    else:
        yield path, value


def tier_numbers() -> None:
    _tier(
        3,
        "numbers",
        "the non-fitting rankers, compared against results_full.json through the driver",
    )
    started = time.perf_counter()
    report = run(cheap_condition(), save=False, verbose=False)
    got = pipeline.render("mutmut", "full", report, None)
    elapsed = time.perf_counter() - started

    if report.undefined:
        raise SystemExit(f"tier 3: {len(report.undefined)} design_point(s) undefined, expected none")

    want = json.loads((config.ARTIFACTS / "results_full.json").read_text())
    measured = set(got["results"]) | set(got["ablations"])
    # The recorded artifact has more conditions than this run, and a low-co-occurrence condition it does not produce.
    # Compare the intersection: the design points this run measured, and every fact about the run.
    for key in ("results", "ablations", "paired_vs_reference", "low_cooccurrence_condition"):
        want[key] = {k: v for k, v in want.get(key, {}).items() if k in measured}
        got[key] = {k: v for k, v in got.get(key, {}).items() if k in measured}

    have, expect = dict(leaves(got)), dict(leaves(want))
    mismatches = [
        f"{path}: got {have[path]!r}, recorded {expect[path]!r}"
        for path in sorted(set(have) & set(expect))
        if have[path] != expect[path]
    ]
    missing = sorted(set(expect) - set(have))
    extra = sorted(set(have) - set(expect))
    print(f"  {len(measured)} rankers, {len(expect)} recorded leaves compared in {elapsed:.1f}s")
    for problem in (mismatches + [f"{p}: missing" for p in missing] + [f"{p}: unexpected" for p in extra])[:20]:
        print(f"  MISMATCH {problem}")
    if mismatches or missing or extra:
        raise SystemExit("tier 3: a recorded number moved")

    # An invariant the artifact cannot state: selection is by rank, so a larger per-change
    # budget selects a superset and recall cannot go down.
    for result in report.design_points:
        recalls = [row["recall"] for row in result.results]
        if recalls != sorted(recalls):
            raise SystemExit(f"tier 3: recall not monotone in budget for {result.design_point.key}: {recalls}")
    print("  recall monotone in budget for every ranker")


def tier_bugsinpy() -> None:
    _tier(4, "real data", "the BugsInPy condition end to end, over real labels")
    result = subprocess.run(
        [sys.executable, "scripts/verify_experiment_layer.py", "bugsinpy"],
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
    )
    lines = [ln for ln in (result.stdout or "").splitlines() if "contrasts" in ln]
    print(f"  {lines[-1].strip() if lines else '(no summary)'}")
    if result.returncode != 0:
        print(result.stdout[-2000:])
        raise SystemExit("tier 4: the BugsInPy condition no longer reproduces")


TIERS = {1: tier_static, 2: tier_unit, 3: tier_numbers, 4: tier_bugsinpy}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="tiers 1-2 only (no dataset needed)")
    parser.add_argument(
        "--with-bugsinpy", action="store_true", help="add tier 4, the real-label condition"
    )
    args = parser.parse_args()

    wanted = [1, 2] if args.quick else [1, 2, 3]
    if args.with_bugsinpy and not args.quick:
        wanted.append(4)

    print("=" * 78)
    print("fast check  --  the full gate is scripts/verify_experiment_layer.py (~30 min)")
    print("=" * 78)
    timings: list[tuple[int, float]] = []
    for number in wanted:
        started = time.perf_counter()
        TIERS[number]()
        timings.append((number, time.perf_counter() - started))

    print("\n" + "=" * 78)
    for number, seconds in timings:
        print(f"  tier {number}: {seconds:6.1f}s")
    print(f"  total   : {sum(s for _, s in timings):6.1f}s")
    not_covered = "the fitted models (XGBoost, SemIf), the low_cooccurrence/covered conditions, the ladder, the variations"
    print(f"  not covered here: {not_covered}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
