# Implementation

Tooling, harness details, and verification results for the RTS feasibility study.
The study design lives in `plan.md`; this document records *how* it is built and
what was measured.

## Environment

- Workspace: `/home/noaha/discriminative_transformer_rts`
- Python: `/home/noaha/graphrnn_env`, Python 3.12.3. Reused deliberately rather than creating a second environment.
- Installed for this work: `mutmut` 3.8.0, `coverage`, `simplejson`. `pytest` 9.0.3 was already present.
- System under test is cloned to `sut/` and gitignored.

## System under test

`marshmallow` at commit `7f0792bd7a06f72e393ec866ac1e89e1502ccfe1` (branch `dev`, 2026-09-15), cloned to `sut/marshmallow`.

Rationale: pure Python, minimal dependency tree (`typing-extensions` only), pytest-based suite, high coverage, and logic-dense validation/serialization code that produces mutants caught by real assertions rather than incidental warnings.

## Mutation framework

`mutmut` 3.8.0. Chosen over `cosmic-ray` because it runs only the tests associated with the mutated function instead of the full suite per mutant, which dominates wall clock. `mutatest` and `MutPy` are unmaintained and excluded. POSIX fork support required.

Accepted limitation: mutmut 3 mutates only inside functions and uses a deliberately narrow operator set (integer +/-1, `<` to `<=`, `break` to `continue`). It cannot produce complex multi-line mutants. Accepted for a feasibility probe; the change-shuffle ablation and the BM25 baseline are what guard against a purely lexical shortcut.

### Configuration

`sut/marshmallow/setup.cfg`, `[mutmut]` section:

- `source_paths = src/marshmallow`
- `pytest_add_cli_args_test_selection = tests/`
- `mutate_only_covered_lines = true`
- `do_not_mutate = *orderedset.py` --- 69% coverage, and it accounts for 17 of the 38 uncovered statements
- `also_copy = conftest.py` --- required, see gotchas below
- `pytest_add_cli_args` deselects three tests --- see gotchas below

Run:

```
cd sut/marshmallow && /home/noaha/graphrnn_env/bin/mutmut run --max-children 8
```

### Harness gotchas (each hit and fixed)

1. **A root-level `conftest.py` is never collected.** mutmut chdirs into its `mutants/` working copy and runs pytest with rootdir `mutants/`, so a conftest at the project root is invisible. Without `also_copy = conftest.py` the outcome hook silently does nothing.
2. **Resolve output paths from the project root, not the cwd.** A cwd-relative path lands in a nested `mutants/mutants/` and is lost when the working copy is regenerated.
3. **`MUTANT_UNDER_TEST` sentinels.** Besides the mutant name, it takes `""`, `mutant_generation`, `fail`, and `stats` during mutmut's own runs. Exclude all of them or the stats run contaminates the outcome table.
4. **State leaks between mutmut's in-process runs.** mutmut runs the stats collection, the clean test, and every mutant worker from a single pytest-importing process, so global state persists across runs. In marshmallow this breaks exactly three tests, on every run after the first:
   - `tests/test_registry.py::test_serializer_class_registry_register_same_classname_different_module`
   - `tests/test_registry.py::test_serializer_class_registry_override_if_same_classname_same_module`
   - `tests/test_schema.py::test_class_registry_returns_schema_type`

   All three assert against the global `marshmallow.class_registry` being pristine. Left in, they fail regardless of the mutant and fabricate "killed" labels for any mutant whose covered functions include `Schema` subclass registration (notably all of `schema.py`). They are deselected via `pytest_add_cli_args`.
5. **Invoke mutmut from the project root.** `mutmut --version` fails elsewhere because config loads at import time. This is not an incompatibility.

### Resolved risks

- `filterwarnings = ["error"]` is **not** set in this project, so warning-triggered kills are not a concern.
- `pytest` 9.0.3 works with mutmut 3.8.0.

## Per-test outcome capture

No mutation framework records per-test outcomes. mutmut stores one exit code per mutant (killed / survived / timeout / no tests / segfault) and carries an explicit TODO for per-test kill attribution. We add it:

- Plugin: `sut/marshmallow/conftest.py`. Hooks `pytest_runtest_logreport`, keys on `MUTANT_UNDER_TEST`, writes `{mutant, nodeid, when, outcome}` to JSONL at session end.
- Output: `mutmut-test-outcomes.jsonl` at the project root (~23 MB). Override with `MUTMUT_OUTCOMES`.
- Caveat: mutmut runs only the tests associated with the mutated function, so a missing (mutant, test) pair means "not selected", not "passed". Absence must never be treated as a pass.

## Coverage baseline (free)

`mutants/mutmut-stats.json` holds `tests_by_mangled_function_name` --- the tests covering each mutated function. Read it directly as the coverage-based RTS baseline rather than shelling out to `mutmut tests-for-mutant`.

## Verification results

- Suite: **1190 tests, all passing, 0.75 s** (three consecutive runs: 0.76 / 0.75 / 0.75 s, no flakiness).
- Coverage: **98%** (1819 statements, 38 missed).
- Mutant population: **2651 mutants** from 13 files. This is the full population, so no subsampling is needed.
- Verdicts: **2311 killed, 340 survived, 0 no-tests, 0 timeout, 0 suspicious** --- mutation score **87.2%**. The zero no-tests count confirms coverage is high enough that every mutant is actionable.
- Wall clock: **79 s** for the entire population with `--max-children 8`, i.e. **37.6 mutants/s**. Cost is not a constraint.
- Test selection is aggressive: a median of **5 tests run per mutant** against a 1190-test suite (~240x reduction). Stats track **188 functions** and **19,390 test-function associations**.
- Cross-validation: the independent `>=1 failing test` signal agrees with mutmut's own killed verdict on **2651/2651 mutants, 0 mismatches**, with 2311 killed by both.
- Artifacts: `mutmut-test-outcomes.jsonl` and `mutants/mutmut-stats.json`.

### Note for the labels

Median killing tests per killed mutant is **1**. Many mutants are caught by exactly one test out of 1190, so per-change recall at small budgets will be low for every model, and a single test's inclusion can flip a mutant from caught to missed. Report the distribution of killing-test counts, not the mean.

## SemIf configuration (pinned)

- Repo: https://github.com/TheoLeeCJ/SemIf-OpenJev (formerly OpenJev). Pin a commit SHA --- the project was renamed recently and is under active development, with no releases published. Model weights are not included; checkpoints come from Hugging Face.
- Mode: `--mode reranker`, not `direct`. Direct mode softmaxes over all options inside a single prompt, so a test's score depends on which other tests were included and is capped at 2-16 options --- unusable for ranking a full suite on one scale. Reranker mode scores each (change, test) pair independently.
- Model: `Qwen/Qwen3-Reranker-4B` at revision `22e683669bc0f0bd69640a1354a6d0aebcfeede5` (BF16, ~9 GB VRAM; CUDA for the reference path, llama.cpp/GGUF for CPU-only).
- Score: raw yes/no relevance log-odds per pair. The repo normalizes log-odds "only for relative comparison"; that normalization destroys comparability across changes, so rank on the unnormalized values.
- Query/document roles: state explicitly which side is the change and which is the test --- rerankers are asymmetric. The codebase renders `Question:` plus `Candidate answer:` into `<Query>` and `state` into `<Document>`, and selects `RETRIEVAL_INSTRUCTION` for rows tagged `code-rag`, the closest existing path to a code-relevance task.
- Input: `--max-tokens 8192` as a ceiling, but pass content-selected inputs (changed function + test function, plus changed-module signatures) rather than truncating at the limit. Report mean tokens per pair; cost scales with actual tokens, not the cap.
- Speed: 1.86 decisions/s measured on a 3090 with no prefix reuse (prefix reuse is a direct-mode feature). Pairs = changes x tests, so ~60k pairs is roughly 8-9 hours. Measure `--pair-batch-sizes` before committing.
- Caveats: returned probabilities are conditional on the supplied options and are not calibrated confidence. Pretraining contamination cannot be ruled out if a well-known project is used.

## Pipeline code

Package `rts/`, run from the workspace root with `/home/noaha/graphrnn_env/bin/python`.

| Module | Role |
|---|---|
| `rts/config.py` | Paths, seed, budgets, pinned SemIf settings |
| `rts/artifacts.py` | Joins mutmut's `.meta` / `.spans` / stats with our outcome log into one `Change` per mutant, including the reconstructed diff |
| `rts/source.py` | Extracts test-function source by pytest node id via `ast` |
| `rts/dataset.py` | Builds the change history, labels, coverage associations, splits, candidate masks |
| `rts/features.py` | Cumulative structured features, BM25 scorer, shuffle transforms |
| `rts/models.py` | All selectors |
| `rts/semif.py` | SemIf pair builder, cost estimator, score cache |
| `rts/evaluate.py` | Per-change budget, metric sweep, paired bootstrap |
| `rts/pipeline.py` | End-to-end orchestration |

Commands:

```
python -m rts.pipeline                                  # full candidate set
python -m rts.pipeline --candidates covered             # SemIf-comparable set
python -m rts.semif --candidates covered                # pairs + cost estimate
python -m rts.semif --candidates covered --build --score # run the reranker
```

### Data model notes

* **Artifact key formats differ and must be reconciled.** Verdict/outcome keys are
  `marshmallow.utils.x_is_generator__mutmut_1`; coverage keys drop the
  `__mutmut_N` suffix; span keys are file-local (`x_is_generator__mutmut_1`).
  Mangled nesting uses `xǁ` rather than `.`, so the last dot reliably separates
  module from function.
* **Change text** comes from the trampoline spans: each function has an
  `__mutmut_orig` block plus one block per mutant. Diffing the two, after
  canonicalizing the trampoline function name, yields the real mutation. All 2651
  changes have a non-empty diff.
* **Imposed temporal order.** Mutants have no intrinsic order, so a fixed-seed
  permutation defines the history. This is why the recency baseline is expected to
  be uninformative, and it is a limitation, not a finding.
* **Negatives.** mutmut runs only the tests associated with the mutated function,
  so a test outside that set is *assumed* not to fail. `Dataset.ran` records which
  tests actually executed, so "passed" and "never ran" stay distinguishable.
* **Candidate sets.** `full` = all 1187 tests. `covered` = only tests covering the
  mutated function, plus that change's killing tests as a safety net. Verified
  invariant: **0 killing tests fall outside the coverage set**, so the `covered`
  mask cannot lose a fault. All selectors are evaluated on the same mask.
* **Budget semantics differ by candidate mode.** A budget is a fraction of *that
  change's own candidate set*, so budget 0.05 means 60 tests in `full` mode but a
  mean of 8 tests in `covered` mode. The two modes are not directly comparable.

## Results

464 held-out faults (of 2311), 530 held-out changes, temporal 80/20 split, 1000
bootstrap resamples. Recall is over fault-bearing changes only.

### Full candidate set (1187 tests per change)

| model | b0.01 | b0.05 | b0.20 |
|---|---|---|---|
| `xgboost_static` (no history) | 0.907 | **0.983** | 1.000 |
| `xgboost_struct_lex` | 0.907 | 0.981 | 0.998 |
| `xgboost_struct` | 0.886 | 0.970 | 1.000 |
| `structural_rule` | 0.539 | 0.800 | 0.976 |
| `coverage` | 0.425 | 0.597 | 0.888 |
| `bm25_lexical` | 0.248 | 0.381 | 0.515 |
| `recency` | 0.093 | 0.381 | 0.804 |
| `failure_rate` | 0.062 | 0.179 | 0.700 |
| `random` | 0.004 | 0.039 | 0.194 |

Paired bootstrap vs `coverage` at budget 0.05 (n=464): `xgboost_struct` +0.373
[+0.325, +0.414], `structural_rule` +0.203 [+0.157, +0.246], `bm25_lexical` -0.216
[-0.272, -0.164], `random` -0.558. All p < 0.0001.

### Covered candidate set (mean 155 tests per change)

| model | b0.01 | b0.05 | b0.20 |
|---|---|---|---|
| `xgboost_struct_lex` | 0.554 | 0.705 | 0.894 |
| `xgboost_static` | 0.541 | 0.692 | 0.888 |
| `xgboost_struct` | 0.472 | 0.631 | 0.862 |
| `failure_rate` | 0.381 | 0.547 | 0.819 |
| `recency` | 0.203 | 0.384 | 0.754 |
| `structural_rule` | 0.179 | 0.289 | 0.504 |
| `bm25_lexical` | 0.203 | 0.269 | 0.429 |
| `random` | 0.052 | 0.073 | 0.239 |
| `coverage` | 0.043 | 0.073 | 0.190 |

## Findings

1. **XGBoost on structured features dominates every non-learned baseline.** At
   budget 0.05 it reaches 0.970 recall against 0.597 for coverage and 0.381 for
   BM25. This is not close.

2. **History features contribute nothing.** `xgboost_static` (cumulative failure
   rate, run count, and recency dropped) matches or beats `xgboost_struct`
   (0.983 vs 0.970 at b0.05). Feature importance agrees: `covers_function`,
   `n_covering_tests`, and `coverage_rank_prior` account for 94.6% of importance,
   and all three history features together for under 3%. So the earlier worry that
   XGBoost was exploiting repeated `(file, test)` pairs is **not** supported — the
   performance survives removing exactly those features.

3. **Most of the achievable recall comes from cheap structural funneling, not from
   semantics.** `covers_function AND module_name_in_test_file` narrows the suite to
   a **median of 9 candidate tests** per change. A hand-built rule using only that
   conjunction plus "shortest test first" scores 0.800 at b0.05, versus 0.970 for
   XGBoost. So the hard part of this task is not ranking; it is the coverage and
   filename funnel, which is non-semantic.

4. **The lexical signal is real but weak, and it is genuinely change-specific.**
   BM25 scores 0.381 at b0.05 (5x the random baseline). The change-shuffle
   ablation collapses it to 0.075 and the test-shuffle to 0.078, so the signal
   comes from matching a specific change to a specific test rather than from a
   change-independent test prior. It is still far below coverage.

5. **The synthetic history massively over-represents repeated `(file, test)`
   pairs.** A `(file, killing-test)` pair recurs a **median of 159 times** (max
   827), because mutation testing revisits the same function (median 8 mutants per
   function). Even the sparsest 5% of changes have their most-repeated pair
   occurring 42 times. Real evolution revisits a function far less often. The
   plan's sparse filter at its natural threshold (1) therefore leaves only 2
   changes; thresholds of 80 and 160 leave 50 and 128 held-out faults. XGBoost
   still scores 0.960-0.980 at b0.05 on those subsets, which is consistent with
   finding 2: the performance is not history-driven.

6. **The task has very little headroom.** With `structural_rule` at 0.800 and
   XGBoost at 0.970-0.983, the space a semantic model could win is roughly 2
   points at b0.05 and essentially zero at b0.20 (XGBoost is at 1.000).

## SemIf status

**Not yet scored.** The adapter, pair builder, cost estimator, and score cache are
implemented and wired into the pipeline as `semif_reranker`; the pipeline skips it
cleanly when the cache is absent. Two things are missing:

1. The SemIf checkout is not installed and `transformers` is not in the
   environment.
2. The checkpoint (`Qwen/Qwen3-Reranker-4B`, ~9 GB) has not been downloaded.

Cost on the `covered` candidate set: **410,167 pairs, ~61 hours (2.6 days)** at the
measured 1.86 decisions/s. This is the binding constraint on the study. Reducing
`--max-changes` is the obvious lever: 300 changes would be ~7 hours.

Comparability is handled by holding the option set constant: every row offers
exactly `yes`/`no` against the same question, so the softmax denominator is
identical across rows and `P(yes)` is on one global scale. Score is the raw log-odds
of `yes` over `no`. See `rts/semif.py` for the reasoning.

## Reproducing

```
cd /home/noaha/discriminative_transformer_rts
python -m rts.artifacts          # change reconstruction summary + sample diff
python -m rts.dataset            # dataset stats + coverage invariant check
python -m rts.features           # feature ranges + BM25 separation
python -m rts.pipeline --bootstrap 1000                 # full candidate set
python -m rts.pipeline --bootstrap 1000 --candidates covered
```

Outputs land in `artifacts/results_{mode}.json` (metrics, ablations, sparse arm,
paired comparisons). Full pipeline runtime is ~70 s, dominated by the two XGBoost
fits (~20 s each).

## Remaining risks

- **Test-selection assumption.** mutmut's selection (coverage plus `max_stack_depth`) assumes tests outside the associated set cannot fail. Spot-check a handful of mutants by running the full suite to confirm the association is not dropping real killers.
- **Pretraining contamination.** Cannot be ruled out for a well-known project; state as a limitation rather than trying to fix it in a probe.
- **Imposed history.** The temporal order is synthetic and the recency baseline is degenerate by construction. Do not report recency as a result.
- **Single revision.** All mutants come from one commit, so there is no real code evolution and no cross-revision drift.

