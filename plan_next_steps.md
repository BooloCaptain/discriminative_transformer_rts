# Plan: the boundary-broken, sparsely-traced regime

Supersedes the previous revision of this file, which was organised around full-suite
relabelling (Gap 1) and a de-lexicalisation ladder (Gap 2). Gap 1 stands. The de-lexicalisation
ladder is **retired** — §0.2 explains why, and the reason is not that it was badly designed but
that it varies the wrong thing.

Study design: `plan.md`. Measured tooling facts: `implementation.md`. This file is the execution
plan for the next phase.

Status: **executed.** W0 (corrected labels), W1 (traceability ladder), W2a (BugsInPy) and the
W2c scoping spike are complete. Results are written up in `implementation.md` §12, and four of
them change the plan's own premises:

* **The ladder confirms the hypothesis, on synthetic labels.** Removing coverage flips SemIf from
  −0.050 behind the best classical selector to **+0.170** ahead (§12.2), and it is *coverage*
  alone — the traceability features add nothing once coverage is gone (L2 ≡ L3).
* **But it does not replicate on real bugs.** On BugsInPy, with an identical feature condition,
  SemIf and BM25 **tie** at every budget (§12.3). This is the most important result here: the
  L3 win looks like an artefact of mutant labels that remain coverage-defined.
* **The boundary does not destroy the lexical bridge.** MicroPython's runner genuinely uses
  `subprocess.Popen`/`pty.openpty()`, yet on 1653 tests BM25 still reaches 0.648 at b0.05 with
  the related test at a median rank of 26/1653 (§12.4). The boundary removes coverage, which
  depends on execution, but not naming, which does not.
* **Real bugs do not bring several failing tests per change.** BugsInPy's 71 usable bugs have a
  median of 1 failing test and a maximum of 4 (§12.3), so the property that motivated W2a is not
  present in it.

The open question has therefore narrowed and changed shape. It is no longer "can SemIf win when
structure is removed" — on synthetic labels it can, on real labels it ties — but "what is
different about the mutant labels that manufactures the win". The natural next step is real
failing-test labels on a boundary-broken corpus (build MicroPython, label its bug commits),
because that is the only arm that changes both the labels and the structure at once.

---

## 0. What changed

### 0.1 The target regime, stated precisely

The deployment being modelled is a test suite that drives an **embedded system across a
boundary** — Python tests sending commands over serial/socket/subprocess to firmware that runs
on a device. Its defining properties:

| property | consequence |
|---|---|
| the changed code does not run in the test process | **coverage cannot cross the boundary**: `covers_function` is empty, or trivially true of a shared driver |
| changed code is firmware (C/C++/Rust/MicroPython), tests are host Python | the lexical bridge is reduced to protocol and feature vocabulary |
| tests are named by scenario, not by module (`test_thermal_shutdown`, not `test_utils.py`) | **filename matching dies** |
| millions of files, thousands of tests, long-running tests | **history is uniformly cold**: no `(file, test)` pair has prior co-occurrence, so failure-history features carry nothing |
| tests take seconds to minutes | budget should be in **seconds**, not test counts |

So four feature families die at once: coverage, filename/path proximity, identifier overlap,
and history. That is the user's hypothesis, and it is mechanically sound: the classical floor in
`implementation.md` is built entirely from those four, so removing them removes the floor.

### 0.2 Why de-lexicalisation is retired

The ladder varied **vocabulary while holding structure fixed**. Measured (§0.6), BM25 collapses
under it (0.381 → 0.125) and the coverage+BM25 tree does not move (0.991 → 0.985), because
coverage carries the result. In the target regime coverage is *absent*, so the ladder's premise
never applies there — the measurement is correct and irrelevant, exactly as the user judged.

The mistake was in the original framing, not the measurement: "sever the lexical bridge" is not
the same manipulation as "remove traceability". The second is what the target regime requires,
and it is what this plan is now organised around.

### 0.3 What the existing evidence already says — and it partly supports the hypothesis

This is worth stating up front, because it is the strongest existing argument for the user's
position and it is already in the data.

In `implementation.md` §5.7, n=141 starved faults, full candidate set, b0.05:

| model | features | b0.01 | b0.05 | b0.10 |
|---|---|---|---|---|
| `xgboost_static_lex` | coverage + BM25 | 0.865 | **0.972** | 0.979 |
| `xgboost_static_nocov_lex` | **BM25 only** — no coverage, no history | 0.326 | 0.582 | 0.723 |
| `bm25_lexical` | BM25 raw, no training | 0.305 | 0.504 | 0.553 |
| **`semif_textonly`** | text only | 0.426 | **0.681** | 0.745 |

**The only place in the whole study where SemIf leads is against the feature-poorest baselines.**
It beats raw BM25 by +0.177 and the BM25-only tree by +0.099 (p=0.015/0.025 at the two tightest
budgets) — and loses to the coverage-bearing tree by −0.291. Remove coverage and SemIf is
already ahead. The direction of the user's hypothesis is visible in the existing numbers.

### 0.4 What the existing evidence says against it — the one thing to settle first

The study's own diagnosis is that SemIf's learned relation is *topical relevance*, while the RTS
relation is *executional and causal*. In the target regime the causal link runs through the
device's behaviour: a firmware change alters what a command does, and a host test fails because
it observed that change. **That link is not written down in either text.** Topical relevance can
only bridge it if the change and the test share *vocabulary about the feature* — a thermal
threshold constant in firmware and `SET_TEMP` in a test are bridgeable; a reordered
initialisation sequence and a timing assertion are not.

So the regime may be one where **the information needed is not in the text at all**, in which
case no text model wins — not the reranker, not BM25 — and the honest answer is that RTS there
needs dynamic analysis or a per-test impact map, not text similarity.

This is cheaply decidable before building anything (§2.3, gate T0): for each fault, measure
whether the killing test shares *any* token with the change text. If the bridge rate is near
zero, both text models are being asked an impossible question and the direction should be
dropped or re-specified. If it is materially non-zero but below the level BM25 can exploit, that
is precisely the regime where a semantic model has room, and the plan is worth executing.

### 0.5 Supporting reconnaissance (already done, still needed)

Two facts about the harness underpin everything below and are not obvious.

**Mutant activation.** mutmut 3.8.0 generates one trampolined working copy in
`sut/marshmallow/mutants/`; each function dispatches on `MUTANT_UNDER_TEST`, read per call. Two
traps: `mutants/src` must precede the editable `marshmallow.pth`, or the tests import the
*unmutated* source and every mutant silently no-ops; and a fresh pytest process per mutant loses
13% of killed mutants, because `tests/conftest.py` builds a schema at import time and a mutant
that raises there aborts the session with zero outcomes. Forking per mutant from a
once-collected parent loses none. Full population: **9.9 min at 8 workers**.

**Full-suite relabelling is complete** (2651 mutants, all gates passing):

| quantity | old labels | full-suite labels |
|---|---|---|
| old killers ⊆ new killers | — | **2651/2651** |
| killers per fault: median / mean / max | 1 / 1.0 / 1 | **8 / 58.4 / 759** |
| faults with exactly one killer | 100% | **13.6%** |
| fault-bearing changes | 2311 | **2327** |
| mutmut survivors that are real faults | 0 | **16/340** |
| faults with an out-of-coverage killer | 0, by construction | **5.5%** (128/2327) |

The documented "median killing tests per killed mutant is 1" is purely an artefact of mutmut's
selection and is now removed. The *circularity* it was meant to fix is smaller than assumed —
5.5% — and is essentially **four `class_registry` tests** that observe global state rather than
the changed function. Artifacts: `artifacts/full_suite_probe_summary.json`,
`artifacts/full_suite_labels.json`; scripts in `scripts/`.

Two harness defects found: a parametrized test whose node id embeds `datetime.now()` and so
changes on every collection, and the three mutmut-deselected tests being absent from the
candidate pool (312 killers across 110 mutants reference unreachable tests).

### 0.6 Retired measurement, kept for the record

| arm | b0.05 | note |
|---|---|---|
| `bm25_lexical` baseline | 0.381 | |
| `bm25_lexical` A1 diff-only obfuscation | 0.125 | vocabulary severed |
| `bm25_lexical` A2 consistent obfuscation | 0.358 | token identity restored |
| `xgboost_static_lex` baseline | 0.991 | |
| `xgboost_static_lex` A1 | 0.985 | **structure unaffected** |
| `xgboost_static_lex` A2 | 0.987 | |

Full candidate set, 464 held-out faults, old labels. The baseline reproduces the documented
0.381, so the harness is faithful.

---

## 1. The target regime as a feature table

This is the organising idea of the plan. Each feature family is either available in the target
regime, absent, or **simulable on marshmallow without any new data**.

| feature family | target regime | marshmallow today | simulable? |
|---|---|---|---|
| `covers_function`, `n_covering_tests`, `coverage_rank_prior` | **absent** (boundary) | present, and dominant | **yes** — drop them; already supported (`exclude_coverage`) |
| `module_name_in_test_file`, `path_distance`, `n_tests_in_file` | **absent** (naming convention differs) | present, and `structural_rule` depends on it | **yes** — needs a new exclusion family (§3) |
| `test_failure_rate_cum`, `test_runs_cum`, `test_last_failure_age` | **absent** (uniformly cold) | present but already harmful under starvation | **yes** — drop globally, not via the starved subset (§3) |
| BM25 over change × test text | **degraded** to protocol/feature vocabulary | present, weak (0.504 raw) | partially — cannot honestly simulate the boundary's vocabulary loss |
| `change_size`, `change_added/removed` | available | present | n/a |
| `test_duration`, `test_n_lines`, `test_n_tokens` | available | present | n/a |
| **coverage-defined candidate mask** (`covered`) | **does not exist**; the pool is the whole suite | most SemIf numbers use it | **yes** — `--candidates full` |
| **labels** | integration-test failures, several per change | co-located unit-test failures, coverage-defined | **no** — this is the real gap |
| candidate-pool size | thousands of tests | 1187 | partially — the bundles study already tripled the pool (158 → 490) |
| budget unit | seconds of wall clock | test count | **yes** — proposal 3's time budgets |

Read the table by column: **everything except the labels is simulable today, for free.** That is
the key planning fact. The traceability-loss ladder is a CPU-only experiment on existing data,
because the SemIf caches are keyed on (change, test) *text* pairs and none of these
manipulations touch the text.

The one thing that cannot be simulated is the label structure, and it is the thing the user
actually cares about. That is why real data is not an optional realism upgrade in this plan — it
is the load-bearing workstream.

---

## 2. Workstreams

### 2.1 W0 — land the corrected labels (prerequisite, ~1.5 h)

Everything downstream is measured against labels, so this comes first. The label set already
exists as a probe artifact; the work is wiring it in.

| step | artifact | cost | status |
|---|---|---|---|
| W0.1 Promote `scripts/probe_full_suite.py` to emit `{mutant, nodeid, when, outcome}` JSONL | `sut/marshmallow/mutmut-full-suite-outcomes.jsonl` | ~1 h | probe done, format to change |
| W0.2 Add the label source + node-id canonicalisation to `rts/config.py`, `rts/artifacts.py` (`--labels {mutmut,full}`) | | ~1 h | not started |
| W0.3 Rebuild the dataset; re-derive the starved masks; re-run the CPU-only arms under both label sets | corrected §5 tables | ~10 min | not started |
| W0.4 Check whether the SemIf side needs re-scoring for the changed fault population | reuse `--exclude-scored` | 0-3 h GPU | not started |

Gates: baseline green (fork, no mutant → `rc=0, 1190 records, 0 failures`) — **PASS**; old killers
⊆ new — **PASS 2651/2651**; `records == 1190` per mutant — **PASS**. Determinism re-run outstanding.

### 2.2 W1 — the traceability-loss ladder (CPU only, ~1 h, no GPU)

**This is the direct test of the user's hypothesis on existing data**, and it is free. It
replaces the retired de-lexicalisation ladder.

Build a monotone ladder of feature availability and evaluate **every selector including SemIf at
every rung**:

| rung | features available | what it models |
|---|---|---|
| L0 | everything | today's benchmark |
| L1 | drop history | uniformly cold history |
| L2 | drop history + coverage | + no cross-boundary coverage |
| L3 | drop history + coverage + name/path proximity | + no naming convention |
| L4 | L3 with no BM25 column | the floor: what survives with no text and no traceability |
| L5 | L3 with BM25 raw only (no trained tree) | SemIf vs bag-of-words, the target comparison |

Two further axes, applied at L3 as the reference rung:

* **candidate pool**: `--candidates full` throughout (the `covered` mask does not exist in the
  target regime), plus a pool-size sweep (1187 → ~3000 by adding non-candidate tests) to model
  thousands of tests.
* **budget in seconds** rather than counts, using a heavy-tailed duration model.

The deliverable is a **degradation curve**, not a single number: recall at b0.05 (and at a
seconds budget) against rung, one line per selector. The question it answers is precise:
*as traceability is removed, does the ordering change, and does SemIf cross the classical
baselines?* On the existing n=141 data the crossing already appears at L2
(`xgboost_static_nocov_lex` 0.582 vs SemIf 0.681); W1 makes that systematic, adds L3–L5, and puts
the corrected labels under it.

**Stated prediction** (so a null is informative): SemIf overtakes the BM25-only tree at L2 and
L3, and its margin grows as rungs are removed. If it does *not*, the hypothesis is falsified on
this SUT.

**The honest caveat, which must appear wherever W1 is reported**: this measures the *degradation
curve*, not the target regime. The labels are still co-located unit-test labels, so the killing
tests are still the tests that cover the changed function. W1 shows what happens when the
*features* go away; it cannot show what happens when the *label structure* is different. Only W2
can.

### 2.3 W2 — real data (the load-bearing workstream)

Three sub-workstreams, in increasing order of value and difficulty.

**W2a — BugsInPy: real bugs and real labels. (~1 week; no GPU for the classical side.)**
493 real bugs across 17 Python projects with known `failing_tests`. This is the cheapest route
to labels that are *not* defined by coverage, which is the study's most consequential
limitation. It also tests two properties the current benchmark cannot: several failing tests per
change, and more than one SUT. Still co-located unit tests, so it does **not** deliver the
boundary — it delivers honest labels and real change text. Needs per-project environments and
test commands. Scoring SemIf is `pairs / 30` seconds (~150k pairs ≈ 83 min for a first pass at
~300 tests per project).

**W2b — real commit history on marshmallow. (~1 h, but read the caveat.)**
The suite is 0.75 s, so it can be run at sampled real revisions. **Correction to the previous
agenda**: this cannot by itself produce faults. Real commits on a healthy project are green, so
there are no failing tests to label. What it buys is realism on the *change* side and on
coverage/history evolution — real diffs, real commit order (making `recency` meaningful), real
test-set churn — and it tests the prediction that real history makes the history features look
weaker, since failures are rare. Treat it as a substrate for W2a-style labels, not as an
experiment.

**W2c — a boundary-structured corpus: the actual target. (Scoping spike first, then a build.)**
This is the only thing that delivers the regime being described, and it is the hardest to source.
It needs a project where (i) the changed code runs behind a process/device boundary, (ii) the
tests drive it, and (iii) bug-inducing commits with per-test outcomes exist.

Candidates, best first:

| candidate | boundary | changed code | labels |
|---|---|---|---|
| **MicroPython** | `run-tests.py` drives a unix port or a device over a pty/serial | C core + Python stdlib | commit history, CI |
| **Zephyr** (Twister + pytest) | pytest drives QEMU or hardware | C firmware | CI, large |
| `esptool` | serial to a device | Python host + stub | mostly mocked — weak |
| CI corpora (TravisTorrent, Bears, GitBug-Java) | varies | varies | **real per-test failure history** |

MicroPython is the leading candidate: its test runner genuinely drives a boundary, the changed
code is genuinely outside the test process, and it has a long real commit history. The scoping
spike is bounded and should answer three questions before any build: is the test runner
deterministic enough to label per-test outcomes; are bug-inducing commits identifiable (SZZ or CI
evidence); and what are the test count and pool size.

**Gate T0 — the textual-bridge audit (§0.4). Do this before the build, not after.** On whatever
corpus W2c selects, measure for each fault whether the killing test shares any token with the
change text, and at what rate the shared tokens are feature/protocol names rather than generic
ones. If the bridge rate is near zero, no text model can work and W2c should be dropped or
re-specified. If it is non-zero but below what BM25 exploits, the regime is exactly the one the
hypothesis predicts, and the build is justified. This is a few hours of work that decides a
multi-week one.

### 2.4 W3 — what cannot be simulated, stated honestly

* **The boundary itself.** On marshmallow, no manipulation of features can make the *labels*
  integration labels. W1 moves features; only W2c moves labels.
* **Cross-language change text.** In the true target the change is firmware and the test is
  Python. marshmallow and BugsInPy are single-language. This cuts both ways: it removes BM25's
  last foothold, and it makes the semantic task much harder for a model trained on NL–PL
  relevance.
* **Hardware coupling, non-determinism, cross-compilation.** Label noise (flipping a small
  fraction of outcomes) remains a cheap partial proxy for flakiness only.

---

## 3. Code changes required

**All done.** Recorded here because the reasoning matters for reading the results:

1. **A feature-exclusion family.** `models.TRACEABILITY_FEATURES = ("filename_stem_match",
   "path_distance", "n_tests_in_file")` now exists. The ladder does not use an exclusion flag;
   it *zeroes* the columns of the removed families, because a zeroed column is exactly "a
   feature that carries no information" and it keeps one code path for the learned and the
   hand-built selectors — so `structural_rule` degrades to "shortest test first" rather than
   crashing, which is the honest behaviour of a method whose input has ceased to exist.
2. **Mislabelled feature fixed.** `STRUCTURED_NAMES[5]` was `"module_name_in_test_file"` but
   held a *filename-stem* match indicator; renamed to `filename_stem_match` across all five
   call sites.
3. **Label source wired.** `config.LABELS` / `RTS_LABELS` selects `mutmut` or `full`, with
   `--labels` on the CLIs. Module state rather than a parameter, because every selector, feature
   and evaluation must agree and threading it through ~20 `dataset.build` call sites would be
   error-prone. `mutmut` stays the default and reproduces every documented number exactly.
4. **Node-id canonicalisation.** `artifacts.canonical_nodeid` collapses the two wall-clock
   parametrization ids that change on every collection. Applied only under `full` labels.
5. **SemIf caches made pool-independent.** `semif.load_scores` now prefers `change_id` /
   `test_nodeid` over `change_row` / `test_col`. The indices are only valid for the exact
   dataset the cache was written against, so a cache silently mis-mapped when the pool grew from
   1187 to 1189.
6. **`--candidates full`** is used for every ladder and BugsInPy number.

---

## 4. Sequencing and costs

```
W0 labels (1.5h CPU)  ──┬─→ W1 ladder (1h CPU, SemIf free)  ──→ degradation curve
                        │
                        └─→ W2a BugsInPy (≈1 week)          ──→ honest labels
                             W2b marshmallow history (1h)   ──→ real change text
                             W2c scoping spike + gate T0 (1 day) ─→ go/no-go
                                   └─→ W2c build (weeks)    ──→ the actual regime
```

W0 first because it is a correctness fix and everything is measured against it. W1 next because
it is nearly free and directly tests the hypothesis on data already in hand — if SemIf does not
cross the BM25-only baseline at L2/L3, the direction is falsified at a cost of one hour, before
any real-data work. Gate T0 runs in parallel with W1, since it is independent.

W2c is the only workstream that reaches the actual target, and it should not start until T0
passes.

---

## 5. Falsifiers

Pre-registered, so a null is informative. Outcomes measured so far are marked.

* **W1 falsifies the hypothesis** if, at L2 and L3, SemIf does not exceed the BM25-only tree on
  the full candidate set with corrected labels. **Not falsified — confirmed.** SemIf leads by
  +0.170 at b0.05 under corrected labels, and the margin is positive at every budget. The
  crossing happens exactly at L2, i.e. when coverage is removed.
* **W1 confirms the hypothesis** if SemIf's margin grows monotonically across L0 → L3. Margin
  goes −0.050 → −0.078 → +0.170 → +0.170 under corrected labels. **Confirmed through L2; L3 adds
  nothing**, so the honest statement is that coverage is the whole effect and filename/path
  proximity is irrelevant. The interesting quantity was the slope, and the slope is a step.
* **Gate T0 kills the direction** if the killing test shares no tokens with the change. **Passes
  weakly on BugsInPy** (87.3% share ≥1 token, but only 2.66 vs 2.03 tokens of lift, and *no*
  lift at all for black and sanic), and **passes strongly on MicroPython** (98.4%). So the task
  is text-solvable in both, which removes the main reason to abandon the direction but also
  undercuts the reason to expect a semantic model to be needed.
* **W2a changes the picture** if real labels move the ordering. **They do — against the
  hypothesis.** On BugsInPy, SemIf and BM25 tie at every budget (0.000 at b0.05, p=1.00), on an
  arm whose feature condition is identical to the ladder's L3, where the ladder reports SemIf
  ahead by +0.170. So the L3 advantage does not replicate once the labels are real. One
  expectation is also refuted: BugsInPy bugs have a median of **one** failing test (max 4), so
  "more killers makes RTS easier" does not apply to it.

---

## 6. Decisions

1. **W1 scope.** Six rungs plus two axes is thorough; a three-rung version (L0, L2, L3) would
   answer the main question in ~20 min. Which?
2. **Pool-size sweep.** Worth simulating "thousands of tests" by inflating the pool, or does the
   bundles study's 158 → 490 result already settle it?
3. **W2 order.** BugsInPy first (honest labels, no boundary) or the W2c scoping spike first (the
   real target, but a go/no-go gate that may kill the direction)?
4. **MicroPython vs Zephyr** for W2c — MicroPython is smaller and has a real serial/pty boundary;
   Zephyr is closer to industrial embedded but is a C build with heavy toolchain needs.
5. **Seconds budget.** Model test durations parametrically, or is the existing `test_duration`
   feature enough to make the axis meaningful?
6. **Doc layout.** Keep this file, or fold it into `plan.md`?
