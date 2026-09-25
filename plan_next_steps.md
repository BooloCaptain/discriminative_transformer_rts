# Plan: executing Gap 1 and Gap 2

Execution plan for the two gaps described in `plan.md` § **Next steps** and
`implementation.md` §11. Study design lives in `plan.md`; measured tooling facts live in
`implementation.md`. This file is the detailed *how* for the next phase and is expected to be
rewritten as the work lands.

Status: **reconnaissance complete.** The relabelling probe has run over the full population and
the de-lexicalisation gate has been measured, so both gaps now have their most important open
question answered before any of the committed work started. Nothing in `rts/` has been changed
yet; the reconnaissance lives in `scripts/probe_full_suite.py`,
`scripts/summarise_full_suite_probe.py`, `artifacts/full_suite_probe_summary.json` and
`artifacts/full_suite_labels.json`. Numbers tagged `[recon]` come from small samples; untagged
numbers are from the full-population run.

Two reconnaissance results change the plan rather than support it, and they are the reason to
read §0.4 and §2.0 before the step tables:

* the label change is much larger than expected, but the *circularity* it was meant to fix is
  much smaller (§0.4);
* de-lexicalisation collapses BM25 but leaves the feature set that actually wins untouched, so
  Gap 2's GPU spend is not justified as specified (§2.0).

---

## 0. What reconnaissance established

The two gaps were specified before anyone had checked that mutmut could be driven the needed
way. It can, but three non-obvious things stand between the spec and a working run, and they
change the implementation.

### 0.1 How a mutant is activated (and the trap)

mutmut 3.8.0 does **not** rewrite files per mutant at run time. It generates one working copy
in `sut/marshmallow/mutants/` in which every function is wrapped by a trampoline, keyed by a
`MutantDict`:

```python
@_mutmut_mutated(mutants_x_is_generator__mutmut)
def is_generator(obj): ...          # trampoline, dispatches on MUTANT_UNDER_TEST
def x_is_generator__mutmut_orig(obj): ...   # original body
def x_is_generator__mutmut_1(obj): ...      # mutant body
```

`wrap_in_trampoline` calls `get_mutant_under_test()` **on every function call**, and reads
`os.environ["MUTANT_UNDER_TEST"]` first. So a mutant can be selected purely from the
environment, per test, without touching the file.

**Trap.** `sut/marshmallow` is installed into `/home/noaha/graphrnn_env` as an editable
install via `site-packages/marshmallow.pth`, which points at the *unmutated* `sut/marshmallow/src`.
Running pytest from `mutants/` therefore imports the original code, every mutant silently
no-ops, and a killed mutant reports `1190 passed`. `mutants/src` must precede that `.pth` on
`sys.path` (mutmut's own `setup_source_paths()` does this by `sys.path.insert(0, ...)`).

Verified:

```
cd sut/marshmallow/mutants
PYTHONPATH=$PWD/src python -c "from marshmallow import utils; \
  print(utils.is_iterable_but_not_string([1,2]))"                       # True
MUTANT_UNDER_TEST=marshmallow.utils.x_is_iterable_but_not_string__mutmut_1 \
  PYTHONPATH=$PWD/src python -c "...same..."                            # False  -> activated
```

### 0.2 Two execution models, and only one works

| model | zero-outcome mutants | verdict |
|---|---|---|
| fresh `pytest` subprocess per mutant | **8/60 `[recon]`** | unusable |
| fork per mutant from a once-collected parent | **0/60 `[recon]`, 0/2651 full run** | use this |

Why the fresh-subprocess model fails: `mutants/tests/conftest.py` and `mutants/tests/base.py`
execute real library code at **import** time (`class UserSchema(Schema)` with
`fields.Str(validate=validate.OneOf(...))`). A mutant whose effect fires there raises during
conftest loading, and a conftest error aborts the entire session — `--continue-on-collection-errors`
does not help, because there is nothing left to collect. Result: zero per-test outcomes, for
13% of killed mutants.

mutmut itself never sees this because its forkserver imports pytest and every test module
once in the parent, then `os.fork()`s per mutant; children inherit `sys.modules` and never
re-execute module-level code. Our runner must do the same: **collect once in a parent, fork
per mutant, run the full suite in the child.**

The fork model is also *more* faithful to the existing labels: children inherit the same
already-imported state mutmut's children had, so import-time mutant effects are excluded
exactly as before. Gap 1 then changes only the test *selection*, which is the point.

### 0.3 Cost, and one flag that matters

- Full suite: **1190 tests**, collection 0.12 s, run ≈0.8 s, no mutant ≈1.2 s wall in a fresh
  process.
- Fork model: **~1.2-1.9 s per mutant** `[recon]`, bounded. The full population (2651
  mutants) took **9.9 min wall at 8 workers**.
- **`--tb=no` is required.** The same mutant that breaks hundreds of tests took 5.4 s with
  tracebacks and 1.5 s without; without it the run is dominated by printing.
- Use `-p no:cacheprovider` (8 children must not race on `.pytest_cache`) and
  `-p no:randomly -p no:random_order` to match mutmut's ordering.
- **Flush `stdout` before forking.** Children inherit the parent's block-buffered stdout and
  re-emit it whenever they flush, which made the probe log 19 MB of duplicated banners. Cosmetic
  but it hides real output, so the production runner should `sys.stdout.flush()`.
- **`os._exit()` discards buffered writes.** A child that writes its result with buffered I/O
  and then calls `os._exit(0)` can lose the file entirely. Flush and `os.fsync` first — this
  cost a debugging cycle during reconnaissance.

### 0.4 What the labels actually change

**Full population, 2651 mutants, executed** (`artifacts/full_suite_probe_summary.json`,
`artifacts/full_suite_labels.json`):

| quantity | old labels | full-suite labels |
|---|---|---|
| old killers ⊆ new killers | — | **2651/2651** |
| mutants with zero recorded outcomes | — | **0** |
| mutants not recording exactly 1190 outcomes | — | **0** |
| fault-bearing changes | 2311 (mutmut's killed) | **2327** |
| killers per fault, median | 1 | **8** (q25 3, q75 33, q95 in tens, max 759) |
| killers per fault, mean | 1.0 | **58.4** |
| total killers | 2311 | **135,848** |
| faults with exactly 1 killer | 2311/2311 (100%) | **317/2327 (13.6%)** |
| faults with an out-of-coverage killer | 0, by construction | **128/2327 (5.5%)** |
| mutmut survivors that are real faults | 0 | **16/340 (4.7%)** |
| killers not in the candidate pool | — | **312** across 110 mutants |

Validation gates 1, 2 and 5 all pass: the fork baseline is `rc=0, 1190 records, 0 failures`;
the old killing set is a subset of the new one for every mutant; and every mutant records
exactly 1190 outcomes.

Three conclusions, and they point in different directions from the framing in `plan.md`.

1. **The killer-count limitation is far worse than documented, and the fix is large.**
   `implementation.md` §2 and §10 both state "median killing tests per killed mutant is 1
   (max 1 across all 2311 faults)". That is entirely an artifact of mutmut running ~5 tests.
   Under full-suite labels a fault has a median of **8** killers and a mean of **58**, and only
   13.6% of faults have exactly one. This removes the §10 limitation "exactly one killing test
   per fault" outright, and per the rung-4 control in §5.6 (where adding killed distractors
   took `random` from 0.090 to 0.360), more killers make RTS *easier*: every selector's recall
   will rise and the margins between them will move. Every headline table needs restating.
2. **The circularity claim is much weaker than `plan.md` assumes.** Out-of-coverage faults are
   **5.5%**, not "structurally excluded". The mask was a small error, not a fatal one, and the
   reflected invariant is *approximately* a discovered property after all.
3. **The out-of-coverage population is not a diverse integration-test family — it is four
   registry tests.** Ranked by how many faults they catch outside coverage:
   `test_multiple_classes_with_all` (114), the two deselected
   `test_serializer_class_registry_*` tests (106, 105), and
   `test_class_registry_returns_schema_type` (101). Every one observes global
   `marshmallow.class_registry` state rather than the changed function. So the integration
   failure mode *is* present, but it is essentially one mechanism, and it is exactly the
   mechanism mutmut's `--deselect` and the `covered` mask were hiding. That makes it cheap to
   characterise and cheap to state honestly, rather than a broad structural blind spot.

The survivor conversion (16/340) is the cleanest measure of what the `covered` mask discarded
outright, because a survivor's killers are out-of-coverage by construction.

**The off-pool killers are a real bug, not cosmetics.** 312 killers across 110 mutants reference
tests the candidate pool cannot select — almost all from the 3 tests missing from
`duration_by_test` (§0.5). Under the `covered` mask they are excluded anyway; under `full` they
would silently cap recall. Decision D3 is therefore load-bearing.

### 0.5 Two harness defects found on the way

- **Time-dependent node ids.** `tests/test_deserialization.py` parametrizes over
  `dt.datetime.now().strftime("%H:%M:%S %Y-%m-%d")` (and the `%m-%d-%Y` form), so **two node
  ids change on every collection**:
  `test_invalid_datetime_deserialization[09-24-2026 13:45:28]` was in the pool built on
  2026-09-24; a collection today yields `[09-25-2026 ...]` instead. Those ids can never be
  looked up in the candidate pool, can never be selected, and are mis-reported as
  out-of-coverage killers. They must be canonicalised to a stable placeholder in the pool,
  the coverage map, and any new labels.
- **Three collected tests are absent from the 1187-test candidate pool.** They are exactly the
  three mutmut deselects in `setup.cfg` (two registry tests and
  `test_schema.py::test_class_registry_returns_schema_type`), which `duration_by_test` never
  recorded. A full-suite run *does* collect and can fail them, so labels would reference tests
  the evaluation can never select. They pass at baseline in the fork model, so the deselect is
  not needed there, but the pool has to be reconciled either way.

The two defects interact, so the arithmetic has to be done once and stated: a fresh collection
reports **1190** node ids, of which the two wall-clock parametrizations collapse to one under
D2, giving **1189** distinct stable tests. The existing pool of 1187 becomes **1186** under the
same canonicalisation, and differs from 1189 by exactly the 3 tests above. Verified:

```
raw pool            : 1187
canonicalised pool  : 1186   (the 2 timestamp ids merge into 1)
fresh collection    : 1190 -> 1189 canonical
1189 - 1186         : 3      (the deselected tests)
```

---

## 1. Gap 1 — full-suite relabelling

**Question it answers.** Do faults exist whose real killer does not cover the changed function,
and how much does the `covered` mask (and the coverage feature that dominates every result)
depend on that being impossible?

### 1.1 Design decisions

| # | decision | choice | rationale |
|---|---|---|---|
| D1 | execution model | fork per mutant from a once-collected parent | §0.2: the only model with zero outcome loss |
| D2 | node ids | canonicalise wall-clock parametrizations to `[<TS>]` in pool, coverage and labels | §0.5; without it, 2 pool ids are dead and 2 tests read as out-of-coverage every run |
| D3 | candidate pool | **1189** distinct stable tests (§0.5), not 1187 | §0.5; a label the pool cannot select caps recall silently |
| D4 | label source | new file + `--labels {mutmut,full}`, **default stays `mutmut`** | keeps every documented number reproducible until the correction is written up; flip the default at the end |
| D5 | muted/import-breaking mutants | none expected (0/150 `[recon]`); record `rc` and zero-record mutants explicitly so they can never be silently dropped | keeps the label set total and auditable |
| D6 | deselects | keep the 3 tests in the full-suite run; do **not** carry mutmut's `--deselect` over | they pass at baseline in the fork model, and two of them are the only genuine out-of-coverage killers found |

### 1.2 Steps

Reconnaissance already produced a working probe and a complete full-population label set as
**interim** artifacts (`artifacts/full_suite_labels.json` in compact index form, plus
`artifacts/full_suite_probe_summary.json`). G1.1 exists to turn that into the production
artifact in the format `rts.artifacts.load_outcomes` already consumes, so nothing downstream
needs a bespoke reader.

| # | step | artifact | cost | status |
|---|---|---|---|---|
| G1.1 | Harden `scripts/probe_full_suite.py` into `scripts/full_suite_labels.py`: emit one JSONL of `{mutant, nodeid, when, outcome}` (resumable), plus a summary; keep the fork model; flush stdout before forking | `sut/marshmallow/mutmut-full-suite-outcomes.jsonl` | ~1 h | probe exists, format to change |
| G1.2 | Run the full population (2651 mutants) | the JSONL above | **9.9 min wall** | done, interim format |
| G1.3 | Validation gates (§1.3) | `artifacts/full_suite_probe_summary.json` | ~5 min | gates 1, 2, 5 pass |
| G1.4 | Add the label source to `rts/config.py` + `rts/artifacts.py` (`load_outcomes(source=...)`, `build_changes(labels=...)`); add D2 canonicalisation to `artifacts.all_test_nodeids` | `rts/artifacts.py`, `rts/config.py` | ~1 h | not started |
| G1.5 | Rebuild the dataset and re-run the **CPU-only** arms under both label sets (`--candidates full` and `covered`), with the starved masks re-derived | corrected tables for §5.1/5.2/5.7 | ~10 min | not started |
| G1.6 | Assess the SemIf side: which held-out changes change fault status, and are their pairs already cached? Score only the missing pairs | reuse `--exclude-scored` | 0-3 h GPU, to be measured | not started |
| G1.7 | Write the correction into `implementation.md` (§2 verified numbers, §3 protocol, §5 tables, §10 limitations, §11 status); add a Gap 1 entry to `rts/variations.py` | docs + `variations.json` | ~1 h | not started |

### 1.3 Validation gates

1. **Baseline is green.** Fork model with no mutant active: expect `rc=0, 1190 records,
   0 failures`. **PASS** (`probe_full_suite.py ... baseline`).
2. **Strict superset.** For all 2651 mutants, the old killing set is a subset of the new one
   after D2 canonicalisation. **PASS 2651/2651**; any miss would have been a bug, not a finding.
3. **Determinism.** Re-run 50 random mutants and require identical failure sets. *Not yet run.*
4. **No spurious registry failures.** The 3 previously-deselected tests must be green for the
   overwhelming majority of mutants, otherwise state leaked into the parent before forking.
   *Indirect evidence only*: 79/80 of a survivor sample produced zero failures, and the fork
   baseline is clean. Worth an explicit check.
5. **Totals reconcile.** `records == 1190` per mutant. **PASS 0 mutants deviating.**
6. **Zero-outcome mutants recorded explicitly** rather than dropped. **PASS: 0 of 2651.**
7. **Killer-count distribution reported**, not just the mean (median, quartiles, max, count of
   single-killer faults). **PASS** — see §0.4.

### 1.4 Deliverables

- The full-suite label set, and a `summary.json` with: faults with ≥1 out-of-coverage killer,
  the "indirect fault" subset (killer exists but no killer covers the changed function), the
  killer-count distribution, and the survivor-to-fault conversion count.
- Corrected headline tables under both label sets, so the study's conclusions can be compared
  before and after.
- An honest structural-funnel ceiling: `structural_rule`'s recall is currently bounded by
  coverage-based narrowing; under new labels the bound changes.
- Updated §10 limitation text and the now-false invariant in `rts/semif.py`'s docstring
  ("Scoring can be restricted to the covered candidate set without losing any fault, because
  every killing test covers the mutated function").

### 1.5 What would falsify the concern

- If out-of-coverage faults were ~0% for both killed and survived mutants, then the `covered`
  mask was near-lossless, the reflected invariant was approximately *discovered* after all, and
  Gap 1 would reduce to the killer-count correction alone. **Measured: 5.5% of faults overall,
  and 16/340 survivors promoted to real faults.** So the concern is real but small in
  incidence and narrow in mechanism; the killer-count correction is the larger part of Gap 1.

### 1.6 Risks

- **Recall rises for everyone.** More killers per fault makes RTS easier. Expect the study's
  main result (SemIf loses) to *strengthen* in direction but compress in margin, and expect
  `random` to improve sharply — §5.6 rung 4 already showed `random` 0.090 → 0.360 when killer
  counts rise. Any table where `random` moves a lot is behaving as predicted, not broken.
- **Cascades are real but noisy.** A mutation in `Field.deserialize` failing 435 tests is
  genuine "everything that goes through this code fails", but it also means one breakage
  dominates. Report the killer distribution and consider a dominance cap as a sensitivity
  analysis (not as the primary label set).
- **The 3 deselected tests are integration tests by nature.** Including them slightly changes
  what the benchmark measures; D6 takes them in deliberately and the count is reported.
- **The starved populations change.** 43 and 141 were keyed on the old failure counts;
  `starved_mask` must be re-derived and the new n reported. The old n=43/141 comparisons become
  non-comparable.

---

## 2. Gap 2 — de-lexicalisation ladder

**Question it answers.** All four text-side levers failed while the shared-vocabulary bridge
between the diff and the tests stayed intact. If that bridge is severed, does BM25 collapse
while SemIf holds?

### 2.0 Reconnaissance already answers the gate, and it changes the design

The plan proposes a cheap CPU gate before spending GPU time. That gate was **run during
reconnaissance**, with an identifier-pseudonym transform (the same construction as arm A1/A2
below) and the current labels:

| selector | baseline | A1 `diff_only` | A2 `consistent` |
|---|---|---|---|
| `bm25_lexical` | 0.248 / **0.381** / 0.459 / 0.515 | 0.047 / **0.125** / 0.164 / 0.216 | 0.244 / **0.358** / 0.438 / 0.498 |
| `xgboost_static_lex` (coverage + BM25) | 0.942 / **0.991** / 0.994 / 1.000 | 0.901 / **0.985** / 0.996 / 1.000 | 0.940 / **0.987** / 0.989 / 1.000 |

(full candidate set, 464 held-out faults, old labels, b0.01/0.05/0.10/0.20; baseline
`bm25_lexical` reproduces the documented 0.381 exactly, so the harness is faithful.)

Predictions 1 and 2 hold: **BM25 collapses under A1** (0.381 → 0.125, close to the 0.039
random floor) and is **almost fully restored under A2** (0.358), which is exactly what
token-identity-preserving obfuscation should do.

But the decisive number is the third row: **the coverage + BM25 tree is essentially immune.**
It moves 0.991 → 0.985 under A1 (−0.006) and 0.987 under A2. De-lexicalisation removes the
signal BM25 alone was using, and the tree does not care, because coverage carries it.

**Consequence for the plan.** The ladder was premised on de-lexicalisation being "the one
manipulation expected to favour a text model". Measured, it does not remove what the classical
side relies on. SemIf sits at ~0.68 on the comparable arm; for the ladder to change the verdict
it would have to gain **~0.30**, while de-lexicalisation costs the classical side 0.006. The
stated prediction ("SemIf drops less but still loses to the coverage + BM25 tree") is therefore
already the answer, and buying ~4.7 h of GPU scoring to confirm it is poor value.

Two options follow, and they should be chosen deliberately rather than by default:

* **Defer Phase B**, and record Gap 2's outcome as: the lexical bridge is *not* what the
  winning features are made of, measured rather than assumed.
* **Re-scope Gap 2** so it attacks what actually carries the result — coverage and names —
  by composing A3 (obfuscate test names and files, killing `name_match_any` and
  `structural_rule`) with proposal 3's *coverage coarsening* (module-granularity coverage,
  dilation with k random coverers, and dropping coverage for a random 50% of tests). That
  combination does degrade the tree, and it is the only manipulation with a real chance of
  moving the verdict. It also needs no SemIf re-scoring for the classical side.

This is a case where reconnaissance saved the expensive step rather than justifying it, which
is the reason the gate exists.

### 2.1 Arms

A deterministic, corpus-wide identifier obfuscation, injected as a text transform on the two
sides of every pair. Pseudonyms are **fixed-length** (`id_0001` style, assigned by first
appearance in sorted order) so length features (`test_n_lines`, `test_n_tokens`, `change_size`)
do not move and the arms differ only in vocabulary.

| arm | change side (diff text) | test side | isolates |
|---|---|---|---|
| **A1 `diff_only`** | identifiers → pseudonyms | unchanged | whether the classical selectors need the diff→test token bridge at all |
| **A2 `consistent`** | identifiers → pseudonyms | identifiers → **same** pseudonyms | whether token *identity* (overlap preserved, names meaningless) is enough; the arm that can most favour a model that reads structure rather than vocabulary |
| **A3 `names_too`** | as A2 | as A2, plus test function/class/file names and node ids obfuscated | whether the *structural* advantage (`name_match_any`, filename matching, `structural_rule`) survives when names carry no information |

An identifier is any `[A-Za-z_][A-Za-z0-9_]*` token that is not a Python keyword, a builtin, or
part of the curated stopword set already in `features.KEYWORDS`. The trampoline `def` name is
already removed by `artifacts.canonicalize()`, so the changed function's own name is *not* in
the diff; the bridge is carried by callees, attributes and locals (`is_generator`,
`hasattr`, `choices`, `self.choices`, ...).

### 2.2 Pre-registered predictions

Stated before running, so a null is informative:

1. **A1 collapses BM25** (expect roughly random, i.e. ≤0.10 at b0.05 from 0.269-0.504) and
   substantially degrades the XGBoost-with-BM25 tree. **Measured: BM25 yes (0.381 → 0.125); the
   tree no (0.991 → 0.985).**
2. **A2 restores most of BM25** (token identity preserved), so BM25(A2) ≈ BM25(baseline).
   **Measured: confirmed (0.358 vs 0.381).** This is the arm that can favour SemIf: if SemIf
   also just matches tokens it is unchanged; if it uses pretrained name semantics it drops.
3. **A3 removes `name_match_any` and most of `structural_rule`**, and degrades the
   coverage+BM25 tree. **Untested — this is now the arm that matters.**
4. **SemIf loses to the coverage+BM25 tree under every arm**, as the semantic hypothesis
   predicts. If it wins under A1 or A2, the hypothesis revives and the finding is large.
   Prediction 1 already makes this near-certain: SemIf would need +0.30.

### 2.3 Implementation hooks

Text already comes from exactly two functions, derived on demand (nothing is stored in
`Dataset`), so the transform is a clean insertion point:

| path | currently | after |
|---|---|---|
| `rts/features.py::change_query_text(change)` | added+removed lines | `lexical.transform_change(change, mode)` |
| `rts/source.py::TestInfo.source` | test function source | `lexical.transform_test(text, mode)` |
| `rts/features.py::build_bm25_scores` | fits over `info.source` | fits over transformed docs and queries |
| `rts/semif_runner.py::build_pair_set` | `features.change_query_text` × `info.source` | transformed, and `mode` in the cache filename |
| `rts/embed.py::build_scores` | same two sources | transformed |
| `rts/models.py::XGBoostSelector` | `ctx.bm25` | unchanged; it consumes `ctx.bm25`, so the transform propagates for free |

New module `rts/lexical.py` (~120 lines): `pseudonym_map`, `transform_text`, `transform_change`,
`transform_nodeid`, `build_transform(mode)`. New experiment `gap2` in `rts/variations.py`,
following the existing `p1`/`p3` pattern, writing into `artifacts/variations.json` and
`artifacts/semif_scores_a{1,2,3}_*.jsonl`.

### 2.4 Staging — the CPU gate comes before any GPU

**Phase A (CPU only, no model, ~10 min).** Compute BM25, `coverage`, `structural_rule`, and the
XGBoost variants under A1/A2/A3 and compare against the baseline, using the *current* label set
so the gates can run before Gap 1 lands. **Already run for A1/A2** (§0.6): BM25 collapses, the
tree does not. What remains is **A3**, which is the arm that actually removes a feature the
tree uses.

**Phase B (GPU).** Re-score SemIf under an arm. Cost at the measured 30.1 pairs/s:

| evaluation subset | pairs | per arm |
|---|---|---|
| starved ≤5 (n≈141), `covered` | ~22k | ~12 min |
| starved ≤5 (n≈141), `full` (1187 candidates) | ~167k | ~93 min |
| all 464 held-out faults, `full` | ~550k | ~5.1 h |

**Revised recommendation: do not queue the three-arm ladder as specified.** Phase A shows the
tree does not move, so Phase B would buy confirmation of a prediction already satisfied. If
the GPU budget is to be spent, spend it on the re-scoped experiment in §2.0 — A3 composed with
coverage coarsening — where the classical side actually degrades and the comparison can go
either way. If it is spent on the original ladder, spend it on **A1 alone** (~93 min), which is
the arm with the most interpretive value.

**Phase C (~10 min).** Paired bootstrap vs `xgboost_static_lex` and `structural_rule` per arm,
plus a lever figure alongside `fig_variation_levers.png`.

### 2.5 Risks

- **A2 may be nearly vacuous for BM25** (that is the prediction), so the informative content of
  the ladder is mostly A1 and A3. Budget accordingly.
- **Cache invalidation is unavoidable.** Any text change invalidates the SemIf cache for that
  arm; there is no way to reuse the existing `semif_scores.jsonl`. This is the same trap P2 hit.
- **Obfuscation interacts with tokenization.** `features.tokenize` lowercases and drops a
  stopword list; pseudonyms must not collide with that list, and `id_` prefixes must be
  consistently cased so `A1`/`A2` differ only in vocabulary.
- **The pair text also feeds `test_n_tokens`.** Fixed-length pseudonyms keep this stable; verify
  by asserting `test_n_tokens` is unchanged between baseline and A2 for every test.

---

## 3. Sequencing

```
Gap 2 Phase A (CPU, ~10 min)  ──────────────┐   no dependency on labels
                                            │
Gap 1 G1.1 → G1.2 → G1.3 (gates) ───────────┤   the correctness fix
                    │                       │
                    └─→ G1.4 → G1.5 → G1.6 → G1.7
                                    │
Gap 2 Phase B (GPU, gated on A) ────┴─→ Phase C
```

Gap 1 goes first in importance: it changes the labels every other number is measured against,
and `--candidates full` is already the documented requirement for any learned-vs-coverage
comparison. Gap 2 Phase A is independent of labels and can run immediately in parallel, which
makes it the natural first thing to execute. Gap 2 Phase B should be run **after** Gap 1 lands,
so its tables are stated on the corrected labels and only one re-scoring pass is needed.

The two gaps compose: the strongest statement available is SemIf vs the coverage+BM25 tree
under de-lexicalisation *and* honest labels. Because a label change moves both sides of every
paired comparison identically, the deltas stay valid; only the levels shift.

---

## 4. Decisions needed before starting

1. **Doc layout.** This file is a third document alongside `plan.md` and `implementation.md`.
   Keep it separate, or fold it into `plan.md` § Next steps?
2. **Candidate pool (D3).** Move to the 1189 distinct stable tests (measured, §0.5), or
   keep 1187 and treat the 3 extra as a documented exclusion? The former is more honest, the
   latter preserves comparability with every published table. Note this is not cosmetic: 312
   killers across 110 mutants currently reference unreachable tests.
3. **Default label source (D4).** Keep `mutmut` as the default until the write-up lands, or
   make `full` the default immediately and mark the old tables as superseded?
4. **Gap 2 re-scoping — the main open call.** Phase A shows the coverage+BM25 tree is immune to
   de-lexicalisation (0.991 → 0.985) while BM25 alone collapses. So: (a) record Gap 2 as a
   measured *negative* for the ladder's premise and spend no GPU; (b) run A1 alone (~93 min) for
   interpretive completeness; or (c) re-scope to A3 + coverage coarsening, which does degrade
   the classical side and is the only version where the verdict could move.
5. **Killer-count sensitivity.** Report the raw full-suite labels as primary, or also a
   dominance-capped variant (e.g. cap each fault at its k most "direct" killers) as the primary?
   With a mean of 58 killers per fault and a max of 759, this choice may matter as much as the
   label change itself.
