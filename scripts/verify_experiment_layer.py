"""Verify the migrated conditions reproduce the recorded artifacts.

The documented numbers are the deliverable (``docs/refactor.md`` §8), so the conditions are checked against
them rather than trusted. Both conditions are checked the same way: run the declared experiment, *render*
the artifact the driver writes, and compare every leaf of the payload.

    study condition (mutmut, full)      vs artifacts/results_full.json
    study condition (mutmut, covered)   vs artifacts/results_coverage_restricted.json
    ladder condition (mutmut)           vs artifacts/ladder.json
    ladder condition (full)             vs artifacts/ladder.json

Comparing the rendered payload rather than the design points is deliberately the stronger test: it
exercises the declarations, the sweep, the metric sweep, the paired contrasts, the dataset
statistics and the renderer at once. A missing leaf is a failure, a differing leaf is a failure,
and an unexpected leaf is a failure unless its path is on the documented addition list below --
so a renamed key cannot pass and a silent extra cannot either.

Run it after any change to the layer or to a condition::

    python scripts/verify_experiment_layer.py                    # everything
    python scripts/verify_experiment_layer.py study              # one condition
    python scripts/verify_experiment_layer.py ladder --labels full

A non-zero exit status means the layer moved a number.

This is the acceptance criterion, and it is slow: about 30 minutes here, almost all of it
fitting XGBoost trees and reading SemIf caches. ``scripts/check_fast.py`` runs the cheap part
of it in about 50 seconds -- the non-fitting rankers and the BugsInPy condition, compared against
the same recorded artifacts through the same renderers -- and prints what it does not cover.
Run that per commit and this before trusting a change to a model, a feature or a cache.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rts import config, studies  # noqa: E402
from rts.render import bugsinpy, ladder, pipeline, variations  # noqa: E402

#: The variation sections the layer produces. ``p5_trained`` and ``p1`` are not migrated -- the
#: first needs an evaluation window inside the held-out tail, the second's cache is absent -- so
#: their recorded sections are left alone rather than compared against something that cannot
#: reproduce them.
VARIATIONS_SECTIONS = (
    "cold_start",
    "cold_start5",
    "cold_start_seeds",
    "p2",
    "p3",
    "p5",
)

FAILURES: list[str] = []
ADDITIONS: list[str] = []
CHECKS = [0]

#: Path prefixes whose extra leaves are expected rather than a regression. One entry, and it is
#: explained where it is produced: the layer pairs *every* model in a group, so it computes a
#: paired delta for the both-shuffled control where the old driver computed one for the two
#: single-shuffled controls only. Dropping it to match the old key set would discard a computed
#: number for cosmetic parity.
ADDITION_PREFIXES = (".paired_vs_reference.",)


def load(name: str) -> dict:
    return json.loads((config.ARTIFACTS / name).read_text())


def leaves(value, path: str = ""):
    """Every leaf of a nested structure, with the path that reaches it."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from leaves(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from leaves(item, f"{path}[{index}]")
    else:
        yield path, value


def compare_payload(label: str, got: dict, recorded: dict) -> None:
    have = dict(leaves(got))
    want = dict(leaves(recorded))
    CHECKS[0] += len(want)
    for path in sorted(set(have) | set(want)):
        if path not in have:
            FAILURES.append(f"{label}{path}: missing (recorded {want[path]!r})")
        elif path not in want:
            if path.startswith(ADDITION_PREFIXES):
                ADDITIONS.append(f"{label}{path} = {have[path]!r}")
            else:
                FAILURES.append(f"{label}{path}: unexpected (got {have[path]!r})")
        elif have[path] != want[path]:
            FAILURES.append(f"{label}{path}: got {have[path]!r}, recorded {want[path]!r}")
    print(f"  {len(want)} recorded leaves compared")


# --- the study condition ---------------------------------------------------------


def verify_study(labels: str = "mutmut", candidate_sets: str = "full") -> None:
    """Run both study conditions and compare the artifact the driver would write."""
    recorded = load(f"results_{candidate_sets}.json")
    print(f"\n--- study condition ({labels}, {candidate_sets}) vs results_{candidate_sets}.json ---")
    # ``save=False``: a check reads the artifacts under test, it does not add to them.
    report, low_cooccurrence_report = studies.run_study(
        labels, candidate_sets=candidate_sets, save=False, verbose=True
    )
    got = pipeline.render(labels, candidate_sets, report, low_cooccurrence_report)
    # The two conditions are separate reports because their budget sets differ -- ``budgets`` is a
    # control -- so ``render`` is what puts them back into one artifact.
    compare_payload(f"study[{labels},{candidate_sets}]", got, recorded)


# --- the ladder condition ---------------------------------------------------------


def verify_ladder(labels: str = "mutmut") -> None:
    recorded = load("ladder.json")[labels]
    print(f"\n--- ladder condition ({labels}) vs ladder.json[{labels!r}] ---")
    compare_payload(f"ladder[{labels}]", ladder.run_label_source(labels, verbose=True), recorded)


def verify_bugsinpy() -> None:
    """Run the real-label condition and compare the artifact it renders.

    This is the one condition whose labels are not defined by coverage, so it is the check that the
    layer reproduces a *measurement* rather than a synthetic construction. Its numbers depend on
    details that are easy to get subtly wrong -- a budget per run, because the recorded
    intervals came from a fresh bootstrap generator each, and a column resolved through each
    bug's own pool -- which is exactly why it is compared leaf by leaf rather than sampled.
    """
    recorded = load("bugsinpy_results.json")
    print("\n--- bugsinpy condition vs bugsinpy_results.json ---")
    got = bugsinpy.run(verbose=False, save=False)
    compare_payload("bugsinpy", got, recorded)


# --- the variation sections -------------------------------------------------


def verify_variations(sections=VARIATIONS_SECTIONS) -> None:
    """Render each migrated variation section and compare it against the recorded one.

    Section by section rather than as one batch, so a section that raises cannot discard the
    contrasts the earlier ones already made -- which matters here because these are the most
    expensive conditions to run.
    """
    recorded = load("variations.json")
    print(f"\n--- variations: {len(sections)} sections vs variations.json ---")
    for section in sections:
        print(f"\n[variations] {section}", flush=True)
        try:
            got = variations.SECTIONS[section]()
        except Exception as exc:  # noqa: BLE001 - a check reports rather than propagates
            FAILURES.append(f"variations[{section}]: raised {exc!r}")
            continue
        compare_payload(f"variations[{section}]", got, recorded[section])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("conditions", nargs="*", choices=["study", "ladder", "variations", "bugsinpy"])
    parser.add_argument("--labels", nargs="*", choices=list(config.LABEL_SOURCES))
    parser.add_argument("--candidate-policy", nargs="*", choices=["full", "coverage_restricted"])
    args = parser.parse_args()

    conditions = args.conditions or ["study", "ladder", "variations", "bugsinpy"]
    labels = args.labels or ["mutmut", "full"]
    candidate_sets = args.candidate_policy or ["full", "coverage_restricted"]

    if "study" in conditions:
        before = CHECKS[0]
        for mode in candidate_sets:
            verify_study("mutmut", mode)
        print(f"  study conditions total: {CHECKS[0] - before} contrasts")
    if "ladder" in conditions:
        before = CHECKS[0]
        for name in labels:
            verify_ladder(name)
        print(f"  ladder conditions total: {CHECKS[0] - before} contrasts")
    if "bugsinpy" in conditions:
        before = CHECKS[0]
        verify_bugsinpy()
        print(f"  bugsinpy condition total: {CHECKS[0] - before} contrasts")
    if "variations" in conditions:
        before = CHECKS[0]
        verify_variations()
        print(f"  variation sections total: {CHECKS[0] - before} contrasts")

    print("\n" + "=" * 78)
    print(f"{CHECKS[0]} contrasts, {len(FAILURES)} mismatch(es)")
    for failure in FAILURES[:40]:
        print(f"  MISMATCH {failure}")
    if len(FAILURES) > 40:
        print(f"  ... and {len(FAILURES) - 40} more")
    if ADDITIONS:
        print(f"\n{len(ADDITIONS)} expected addition(s) beyond the recorded artifact:")
        for addition in ADDITIONS[:8]:
            print(f"  ADDED {addition}")
        if len(ADDITIONS) > 8:
            print(f"  ... and {len(ADDITIONS) - 8} more")
    print("=" * 78)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
