"""Verify the migrated arms reproduce the recorded artifacts.

The documented numbers are the deliverable (``refactor.md`` §8), so the arms are checked against
them rather than trusted. Both arms are checked the same way: run the declared experiment, *render*
the artifact the driver writes, and compare every leaf of the payload.

    study arm (mutmut, full)      vs artifacts/results_full.json
    study arm (mutmut, covered)   vs artifacts/results_covered.json
    ladder arm (mutmut)           vs artifacts/ladder.json
    ladder arm (full)             vs artifacts/ladder.json

Comparing the rendered payload rather than the cells is deliberately the stronger test: it
exercises the declarations, the sweep, the metric sweep, the paired comparisons, the dataset
statistics and the renderer at once. A missing leaf is a failure, a differing leaf is a failure,
and an unexpected leaf is a failure unless its path is on the documented addition list below --
so a renamed key cannot pass and a silent extra cannot either.

Run it after any change to the layer or to an arm::

    python scripts/verify_experiment_layer.py                    # everything
    python scripts/verify_experiment_layer.py study              # one arm
    python scripts/verify_experiment_layer.py ladder --labels full

A non-zero exit status means the layer moved a number.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rts import config, ladder, pipeline, studies  # noqa: E402

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


# --- the study arm ---------------------------------------------------------


def verify_study(labels: str = "mutmut", candidates: str = "full") -> None:
    """Run both study arms and compare the artifact the driver would write."""
    recorded = load(f"results_{candidates}.json")
    print(f"\n--- study arm ({labels}, {candidates}) vs results_{candidates}.json ---")
    # ``save=False``: a check reads the artifacts under test, it does not add to them.
    report, sparse_report = studies.run_study(
        labels, candidates=candidates, save=False, verbose=True
    )
    got = pipeline.render(labels, candidates, report, sparse_report)
    # The two arms are separate reports because their budget sets differ -- ``budgets`` is a
    # knob -- so ``render`` is what puts them back into one artifact.
    compare_payload(f"study[{labels},{candidates}]", got, recorded)


# --- the ladder arm ---------------------------------------------------------


def verify_ladder(labels: str = "mutmut") -> None:
    recorded = load("ladder.json")[labels]
    print(f"\n--- ladder arm ({labels}) vs ladder.json[{labels!r}] ---")
    compare_payload(f"ladder[{labels}]", ladder.run_label_source(labels, verbose=True), recorded)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arms", nargs="*", choices=["study", "ladder"])
    parser.add_argument("--labels", nargs="*", choices=list(config.LABEL_SOURCES))
    parser.add_argument("--candidates", nargs="*", choices=["full", "covered"])
    args = parser.parse_args()

    arms = args.arms or ["study", "ladder"]
    labels = args.labels or ["mutmut", "full"]
    candidates = args.candidates or ["full", "covered"]

    if "study" in arms:
        before = CHECKS[0]
        for mode in candidates:
            verify_study("mutmut", mode)
        print(f"  study arms total: {CHECKS[0] - before} comparisons")
    if "ladder" in arms:
        before = CHECKS[0]
        for name in labels:
            verify_ladder(name)
        print(f"  ladder arms total: {CHECKS[0] - before} comparisons")

    print("\n" + "=" * 78)
    print(f"{CHECKS[0]} comparisons, {len(FAILURES)} mismatch(es)")
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
