"""Probe: run the FULL 1190-test suite for a sample of mutants (Gap 1 feasibility).

Gap 1 in ``plan.md`` is *full-suite relabelling*: mutmut only ever runs the tests that
cover the mutated function (median 5 of 1190), so a fault whose real killer lies outside
that set is recorded as "never ran" and treated as not failing. This probe replaces that
selection step with the whole suite and reports what changes.

Execution model (verified against mutmut 3.8.0)
----------------------------------------------
mutmut activates a mutant by setting ``MUTANT_UNDER_TEST`` and reading it inside a
trampoline on every call, so a mutant can be selected from the environment. Two ways to
run the full suite under that env var:

* **Fresh subprocess per mutant** --- simple, but re-executes conftest/module-level code.
  ``tests/conftest.py`` and ``tests/base.py`` build ``UserSchema`` at *import* time, so any
  mutant whose effect fires there (e.g. ``OneOf.__init__``) aborts the whole session and
  yields **zero** per-test outcomes. Measured: 8/60 killed mutants (13%) are lost this way.
* **fork-per-mutant from a once-collected parent** --- replicates mutmut's own model
  (import once, fork per mutant), so children inherit ``sys.modules`` and never re-execute
  module-level code. Measured: 0/60 lost.

This probe uses the fork model. Two further requirements discovered the hard way:

* ``mutants/src`` must precede the editable install's ``marshmallow.pth``, or the tests
  import the *unmutated* SUT source and every mutant silently no-ops.
* ``--tb=no`` bounds the worst case: a mutant that breaks hundreds of tests took 5.4s with
  tracebacks and 1.5s without.

Known wart: every forked child inherits the parent's block-buffered ``stdout``, so when a
child flushes (pytest does) it re-emits everything the parent had buffered. The log therefore
repeats the banner once per child. It is cosmetic here, but the production runner should
``sys.stdout.flush()`` before forking.

Modes
-----
``killed`` / ``survived`` / ``all``  sample the mutant population; ``baseline`` forks a single
child with no mutant active, which is validation gate 1 (expect 1190 collected, 0 failed).

Usage
-----
    python scripts/probe_full_suite.py [N] [WORKERS] [killed|survived|all|baseline]
"""

from __future__ import annotations

import gc
import json
import os
import random
import re
import sys
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
SUT = WORKSPACE / "sut" / "marshmallow"
MUTANTS = SUT / "mutants"
OLD_OUTCOMES = SUT / "mutmut-test-outcomes.jsonl"
OUT_DIR = WORKSPACE / "artifacts" / "probe_full_suite"

PYTEST_ARGS = [
    "-q",
    "--tb=no",
    "-p",
    "no:cacheprovider",
    "-p",
    "no:randomly",
    "-p",
    "no:random_order",
]

MUTANT_SUFFIX_RE = re.compile(r"__mutmut_(\d+|orig)$")

# Sentinel used by the ``baseline`` mode: no mutant is activated.
BASELINE = "__baseline__"

# ``tests/test_deserialization.py`` parametrizes over ``dt.datetime.now()``, so two
# nodeids embed the collection wall-clock and change on EVERY fresh collection:
#   test_invalid_datetime_deserialization[13:45:28 2026-09-24]
#   test_invalid_datetime_deserialization[09-24-2026 13:45:28]
# The candidate pool was built from a 2026-09-24 collection, so a re-collection produces
# ids that can never be looked up, can never be selected, and would be mis-reported as
# out-of-coverage killers. Canonicalise them back to the pool's placeholder.
_TS_PARAMS = (
    re.compile(r"\[\d{2}:\d{2}:\d{2} \d{4}-\d{2}-\d{2}\]"),
    re.compile(r"\[\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2}\]"),
)


def canonical_nodeid(nodeid: str) -> str:
    out = nodeid
    for pat in _TS_PARAMS:
        out = pat.sub("[<TS>]", out)
    return out


class Recorder:
    """Per-test outcome collector, installed as a plugin into each forked session."""

    def __init__(self) -> None:
        self.records: list[dict[str, str]] = []

    def pytest_runtest_logreport(self, report) -> None:
        if report.when == "call" or report.outcome == "failed":
            self.records.append(
                {"nodeid": report.nodeid, "when": report.when, "outcome": report.outcome}
            )


def load_verdicts() -> dict[str, int | None]:
    verdicts: dict[str, int | None] = {}
    for path in sorted((MUTANTS / "src").rglob("*.py.meta")):
        verdicts.update(json.loads(path.read_text())["exit_code_by_key"])
    return verdicts


def load_coverage() -> dict[str, list[str]]:
    stats = json.loads((MUTANTS / "mutmut-stats.json").read_text())
    raw = stats["tests_by_mangled_function_name"]
    return {k: [canonical_nodeid(t) for t in v] for k, v in raw.items()}


def load_pool() -> set[str]:
    """The evaluation candidate pool, canonicalised to match a fresh collection."""
    stats = json.loads((MUTANTS / "mutmut-stats.json").read_text())
    return {canonical_nodeid(t) for t in stats["duration_by_test"]}


def load_old_killers() -> dict[str, set[str]]:
    """The existing (mutmut-selected) labels, for the subset check."""
    out: dict[str, set[str]] = {}
    with OLD_OUTCOMES.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["outcome"] == "failed":
                out.setdefault(rec["mutant"], set()).add(rec["nodeid"])
    return out


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    mode = sys.argv[3] if len(sys.argv) > 3 else "killed"
    global OUT_DIR
    if len(sys.argv) > 4:
        OUT_DIR = Path(sys.argv[4])
    elif os.environ.get("PROBE_OUT_DIR"):
        OUT_DIR = Path(os.environ["PROBE_OUT_DIR"])

    os.chdir(MUTANTS)
    sys.path.insert(0, str(MUTANTS / "src"))
    os.environ.pop("MUTANT_UNDER_TEST", None)
    # Stop the repo's conftest plugin writing its own outcome log during this probe.
    os.environ["MUTMUT_OUTCOMES"] = os.devnull

    import pytest

    t0 = time.time()
    rc = pytest.main(["--collect-only", "-q", "-p", "no:cacheprovider"])
    print(f"parent collect-only: rc={rc} in {time.time() - t0:.1f}s")
    gc.freeze()

    verdicts = load_verdicts()
    old = load_old_killers()
    if mode == "survived":
        population = [m for m, e in verdicts.items() if e == 0]
    elif mode == "all":
        population = list(verdicts)
    elif mode == "baseline":
        # Validation gate 1: no mutant active, so every trampoline calls its original.
        # Must be 1190 collected / 0 failed, using the same fork path as a real mutant.
        population = [BASELINE]
    else:
        population = [m for m, e in verdicts.items() if e in (1, 3)]

    random.seed(7)
    n = min(n, len(population))
    sample = population if mode == "baseline" else random.sample(population, n)
    print(f"mode={mode} population={len(population)} sample={len(sample)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    running: dict[int, tuple[str, float]] = {}
    results: list[tuple[str, float, dict]] = []

    def spawn(mutant: str) -> None:
        pid = os.fork()
        if pid != 0:
            running[pid] = (mutant, time.time())
            return
        rc = -1
        try:
            if mutant == BASELINE:
                os.environ.pop("MUTANT_UNDER_TEST", None)
            else:
                os.environ["MUTANT_UNDER_TEST"] = mutant
            rec = Recorder()
            rc = int(pytest.main(["tests/"] + PYTEST_ARGS, plugins=[rec]))
            failures = sorted(
                {canonical_nodeid(r["nodeid"]) for r in rec.records if r["outcome"] == "failed"}
            )
            payload = {
                "mutant": mutant,
                "rc": rc,
                "records": len(rec.records),
                "failures": failures,
            }
        except BaseException:
            payload = {"mutant": mutant, "rc": rc, "records": 0, "failures": []}
        try:
            path = OUT_DIR / f"{abs(hash(mutant))}.json"
            with path.open("w") as fh:
                fh.write(json.dumps(payload))
                fh.flush()
                os.fsync(fh.fileno())
        finally:
            os._exit(0)
    pending = list(sample)
    t_start = time.time()
    while pending or running:
        while pending and len(running) < workers:
            spawn(pending.pop())
        pid, _status = os.wait()
        mutant, t0m = running.pop(pid)
        path = OUT_DIR / f"{abs(hash(mutant))}.json"
        data = (
            json.loads(path.read_text())
            if path.exists()
            else {"records": 0, "failures": [], "rc": -1}
        )
        results.append((mutant, time.time() - t0m, data))
    wall = time.time() - t_start

    secs = sorted(r[1] for r in results)
    print(f"\nwall: {wall:.1f}s for {len(results)} mutants, {workers} workers")
    print(
        f"per-mutant: min {secs[0]:.2f} median {secs[len(secs) // 2]:.2f} max {secs[-1]:.2f}"
    )

    zero = [r for r in results if r[2]["records"] == 0]
    print(f"ZERO-outcome: {len(zero)}/{len(results)}")
    for m, _, d in zero[:5]:
        print(f"    {m} rc={d['rc']}")

    subset_ok = 0
    misses = []
    for mutant, _, d in results:
        o = {canonical_nodeid(t) for t in old.get(mutant, set())}
        if o <= set(d["failures"]):
            subset_ok += 1
        else:
            misses.append((mutant, sorted(o)[:2], d["failures"][:3]))
    print(f"old killers subset of new: {subset_ok}/{len(results)}")
    for m, o, nw in misses[:6]:
        print(f"    MISS {m}: old={o} new={nw}")

    if mode == "baseline":
        b = results[0][2]
        print(f"BASELINE rc={b['rc']} records={b['records']} failures={len(b['failures'])}")
        for f in b["failures"][:10]:
            print(f"    UNEXPECTED FAIL {f}")
        print("gate 1 PASS" if (b["rc"] == 0 and not b["failures"]) else "gate 1 FAIL")
        return

    tot_old = sum(len({canonical_nodeid(t) for t in old.get(m, set())}) for m, _, _ in results)
    tot_new = sum(len(d["failures"]) for _, _, d in results)
    print(f"total killers old/new: {tot_old} / {tot_new}")

    # The headline: do killers lie outside the mutated function's coverage set?
    cov = load_coverage()
    pool = load_pool()
    fault_bearing = [d for _, _, d in results if d["failures"]]
    out_of_cov = []
    off_pool_total = 0
    off_pool_mutants = 0
    for d in fault_bearing:
        key = MUTANT_SUFFIX_RE.sub("", d["mutant"])
        covering = set(cov.get(key, []))
        off_pool = [t for t in d["failures"] if t not in pool]
        off_pool_total += len(off_pool)
        if off_pool:
            off_pool_mutants += 1
        if not set(d["failures"]) <= covering:
            out_of_cov.append(d)
    print(f"fault-bearing in sample            : {len(fault_bearing)}")
    print(f"  with an OUT-OF-COVERAGE killer   : {len(out_of_cov)} ({len(out_of_cov)/max(len(fault_bearing),1):.1%})")
    print(f"  killers NOT in candidate pool    : {off_pool_total} across {off_pool_mutants} mutants")
    for d in out_of_cov[:8]:
        key = MUTANT_SUFFIX_RE.sub("", d["mutant"])
        covering = set(cov.get(key, []))
        outside = sorted(set(d["failures"]) - covering)
        print(f"    {d['mutant']}")
        print(f"        covering={len(covering)} killers={len(d['failures'])} outside={outside[:3]}")

    est = 2651 * (sum(secs) / len(secs)) / workers / 60
    print(f"projected full run: {est:.1f} min wall at {workers} workers")


if __name__ == "__main__":
    main()
