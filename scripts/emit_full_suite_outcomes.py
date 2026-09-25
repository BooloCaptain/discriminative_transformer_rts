"""Emit the full-suite relabelling in the format ``rts.artifacts`` already consumes.

Input:  ``artifacts/full_suite_labels.json`` -- the compact probe artifact, which stores
        per mutant the *indices* of the tests that failed plus the canonical test list.
Output: ``sut/marshmallow/mutmut-full-suite-outcomes.jsonl`` -- one record per (mutant, failing
        test), in the same shape as the mutmut-selected log, so ``load_outcomes`` needs no
        bespoke reader.
        ``sut/marshmallow/mutmut-full-suite-tests.json`` -- the canonical candidate pool.

Why only failures are written
-----------------------------
In the full-suite design every collected test runs against every mutant, so "ran" is
trivially all tests and the complement of the failure list is "passed". Writing 3.15M passing
records would be ~350 MB of I/O and a multi-GB nested dict at load time for no information.
``artifacts.build_changes`` reconstructs ``ran`` as the full pool when ``RTS_LABELS=full``.

Both outputs land in ``sut/``, which is gitignored.

Usage
-----
    python scripts/emit_full_suite_outcomes.py [--labels artifacts/full_suite_labels.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
SUT = WORKSPACE / "sut" / "marshmallow"
DEFAULT_IN = WORKSPACE / "artifacts" / "full_suite_labels.json"
OUT_RECORDS = SUT / "mutmut-full-suite-outcomes.jsonl"
OUT_TESTS = SUT / "mutmut-full-suite-tests.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_IN)
    args = parser.parse_args()

    data = json.loads(args.labels.read_text())
    test_ids: list[str] = data["test_ids"]
    mutants: dict[str, dict] = data["mutants"]
    print(f"[emit] {len(mutants)} mutants, {len(test_ids)} canonical tests")

    n_records = 0
    with OUT_RECORDS.open("w") as fh:
        for mutant, rec in mutants.items():
            for idx in rec["failures"]:
                fh.write(
                    json.dumps(
                        {
                            "mutant": mutant,
                            "nodeid": test_ids[idx],
                            "when": "call",
                            "outcome": "failed",
                        }
                    )
                    + "\n"
                )
                n_records += 1

    OUT_TESTS.write_text(
        json.dumps(
            {
                "test_ids": test_ids,
                "n_tests": len(test_ids),
                "n_mutants": len(mutants),
                "mutants": sorted(mutants),
                "source": str(args.labels.relative_to(WORKSPACE)),
                "note": (
                    "Canonical candidate pool for full-suite labels: wall-clock parametrizations "
                    "collapsed to [<TS>], and the three mutmut-deselected tests included. "
                    "'mutants' is the set that was actually run, so a mutant with no failure "
                    "records is a survivor rather than a missing run."
                ),
            },
            indent=2,
        )
    )

    print(f"[emit] wrote {n_records:,} failure records to {OUT_RECORDS.relative_to(WORKSPACE)}")
    print(f"[emit] wrote pool of {len(test_ids)} to {OUT_TESTS.relative_to(WORKSPACE)}")
    zero = sum(1 for r in mutants.values() if not r["failures"])
    print(f"[emit] {zero} mutants have no failing test (survivors under full-suite labels)")


if __name__ == "__main__":
    main()
