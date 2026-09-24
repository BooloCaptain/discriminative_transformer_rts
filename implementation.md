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
   (0.983 vs 0.970 at b0.05), so the earlier worry that XGBoost was exploiting
   repeated `(file, test)` pairs is **not** supported. Note this contradicts the
   feature importances, which put ~0.18 on the history features in the bundle
   setup: that importance is redundant signal, not necessary signal. Ablations are
   the reliable instrument here, not importances.

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

7. **But in a data-starved history, SemIf overtakes XGBoost.** Filtering to
   changes whose killing `(file, test)` pair has essentially no failure history
   (the proxy for a huge codebase where a file has not changed in years), SemIf
   reaches 0.442 at b0.05 against 0.256 for the best XGBoost, and leads at every
   budget. The effect is monotone in starvation (XGBoost 2.1x ahead on the full
   set, 1.2x on `failures <= 5`, 1.7x *behind* on `failures <= 2`), and the
   mechanism is coherent: the history baselines collapse to near-random while
   SemIf's relative recall *rises* as history is removed. The margins are marginal
   under multiple-comparison correction and n is small (43), so see
   **Starved regime** below for the caveats before citing this.

So the answer to the study's question is not a flat no. **XGBoost wins comfortably
on a history-rich benchmark, but SemIf wins in the starved regime that the study
was actually built to probe.** Both halves are needed to state the result honestly.

## SemIf status

**Setup is complete and the model runs.** Scoring the full grid has not been done.

### Environment

SemIf installed in-place into `graphrnn_env` (user's choice) at pinned commit
`23cf1f39fc9534fe81437200959b6dfc7106e45a`. A pre-install snapshot is at
`/home/noaha/graphrnn_env_before_semif.txt` (159 packages) if a rollback is needed.

What changed:

| | before | after |
|---|---|---|
| torch | 2.10.0+rocm7.1 | **unchanged** (`==2.10.0` matched the local build) |
| numpy | 2.3.5 | 2.2.6 |
| protobuf | 6.33.5 | 7.36.1 |
| typing-extensions | 4.15.0 | 4.16.0 |
| new | — | transformers 5.17.0, tokenizers 0.23.2, safetensors 0.8.0, huggingface-hub 1.31.0, accelerate 1.12.0, sentencepiece 0.2.1, typer, httpx |

**ROCm survived**: `torch.cuda.is_available()` is still True on the RX 9070 XT.
The RTS pipeline still reproduces identical numbers after the numpy downgrade
(0.039 / 0.381 / 0.179 / 0.597 at budget 0.05), so the downgrade is benign for
this study.

`[test]` was deliberately **not** installed: it pins `pytest==8.4.2`, which would
downgrade pytest 9.0.3 and could break the mutmut harness.

Checkpoint: `Qwen/Qwen3-Reranker-4B` @ `22e683669bc0f0bd69640a1354a6d0aebcfeede5`,
7.6 GB, cached at `/home/noaha/hf_cache` (set `HF_HOME` to use it). Loads in ~8 s
and fits in 17 GB BF16.

### Scorer

`rts/semif_runner.py` builds the prompt directly from the model's native template
and imports SemIf's `PREFIX`, `SUFFIX`, and `_answer_ids` so the contract stays
identical to the reference implementation. Pairs are scored independently with the
same prompt skeleton and read out as the yes-vs-no log-odds, so scores are on one
global scale across changes. Pairs are batched (left-padded) in one forward pass.

`orientation` selects which side is the Query and which the Document, since
rerankers are asymmetric.

### Preliminary smoke test (n=10, not conclusive)

Ranking a known killing test against 9 distractors:

| setup | #1 | mean rank | chance |
|---|---|---|---|
| distractors from the coverage set | 2/10 | — | 1.00 |
| distractors random from the full suite | 3/10 | — | 1.00 |
| same, `change`=Query | 4/10 | 3.40 | 5.50 |
| same, `test`=Query | 6/10 | 3.80 | 5.50 |

Both orientations beat chance but neither is strong, and with n=10 the difference
between them is within noise, so the full pass used `change_query`.

### Measured throughput (the cost objection is resolved)

**22.3-22.8 pairs/s at batch 16** (421 padded tokens/pair), versus the 1.86
decisions/s the repo reports for batch-1 native reranker scoring on a 3090. Batching
gives a **12x speedup**, which changes the budget entirely:

| scope | pairs | at 1.86/s (assumed) | **measured** |
|---|---|---|---|
| Full grid | 410k | 61 h | ~5 h |
| **Held-out only** | 78k | 12 h | **~1.0 h** |
| Held-out + change-shuffle | 157k | 24 h | ~2 h |

So the full held-out pass is a one-hour job, not a multi-day one. The earlier
"61 h" figure in this document was based on the repo's batch-1 number and is
superseded.

### Full held-out result, text-only arm (464 faults, covered candidates)

| model | b0.01 | b0.05 | b0.10 | b0.20 |
|---|---|---|---|---|
| `xgboost_struct_lex` | 0.554 | **0.705** | 0.823 | 0.894 |
| `xgboost_static` | 0.541 | 0.692 | 0.791 | 0.888 |
| `xgboost_static_nocov_lex` | 0.489 | 0.655 | 0.750 | 0.869 |
| `xgboost_struct` | 0.472 | 0.631 | 0.759 | 0.862 |
| `xgboost_static_nocov` | 0.418 | 0.569 | 0.679 | 0.823 |
| `failure_rate` | 0.381 | 0.547 | 0.681 | 0.819 |
| `recency` | 0.203 | 0.384 | 0.556 | 0.754 |
| **`semif_textonly`** | **0.226** | **0.306** | **0.377** | **0.511** |
| `structural_rule` | 0.179 | 0.289 | 0.375 | 0.504 |
| `bm25_lexical` | 0.203 | 0.269 | 0.336 | 0.429 |
| `random` | 0.052 | 0.073 | 0.142 | 0.239 |
| `coverage` | 0.043 | 0.073 | 0.121 | 0.190 |

Paired bootstrap vs `xgboost_struct` at b0.05 (n=464):
`semif_textonly` -0.325 [-0.390, -0.265] p<0.0001; `bm25_lexical` -0.362
[-0.425, -0.302]; `structural_rule` -0.343 [-0.394, -0.293]; `failure_rate`
-0.084 [-0.134, -0.039] p=0.002; `xgboost_static` +0.060 [+0.026, +0.093].

SemIf text-only vs BM25: **+0.037 [-0.002, +0.080], p=0.068** -- not significant.

### What the text-only arm establishes

1. **SemIf with raw text is statistically indistinguishable from BM25** (p=0.068)
   and from a three-line structural rule. Giving a 4B reranker the diff and the
   test source buys nothing over bag-of-words.
2. **A tree model with no coverage, no history, and no text at all still beats it
   by ~1.9x** (`xgboost_static_nocov` 0.569 vs 0.306). That is the cleanest
   statement of the result: cheap static structure dominates raw-text semantic
   reading on this task.
3. **`xgboost_static_nocov_lex` (0.655) shows the lexical signal is real but is
   used far better as a feature than as a prompt.** It is statistically
   indistinguishable from the full `xgboost_struct` (-0.024, p=0.33) while using
   no coverage and no history.

### Caveat on the comparison

SemIf has raw text; XGBoost has BM25 plus structured features. Neither is a strict
superset of the other. But the two are now much closer to like-for-like than the
earlier text-only-vs-structure-only framing, and the direction is unambiguous.
The mirror arm (all 15 features *plus* text) is the arm that removes the remaining
asymmetry.

### Pilot vs baselines, same 60 held-out faults (superseded)

Kept for the record. The full 464-fault run above supersedes these numbers, and
this table is why the fairness question mattered: at n=60 SemIf looked tied with
`recency` and below `structural_rule`, but on the full set it edges both.

Recall at each budget, all selectors evaluated on exactly the rows the SemIf pilot
scored, using the `covered` candidate set:

| model | b0.01 | b0.05 | b0.10 | b0.20 |
|---|---|---|---|---|
| `xgboost_static` | 0.550 | **0.733** | 0.800 | 0.883 |
| `xgboost_struct_lex` | 0.517 | 0.717 | 0.833 | 0.883 |
| `xgboost_struct` | 0.450 | 0.650 | 0.800 | 0.850 |
| `failure_rate` | 0.300 | 0.500 | 0.667 | 0.800 |
| `recency` | 0.217 | 0.317 | 0.467 | 0.733 |
| `structural_rule` | 0.250 | 0.333 | 0.400 | 0.533 |
| **`semif_reranker`** | **0.200** | **0.317** | **0.383** | **0.483** |
| `bm25_lexical` | 0.183 | 0.250 | 0.333 | 0.400 |
| `random` | 0.067 | 0.083 | 0.133 | 0.250 |
| `coverage` | 0.050 | 0.067 | 0.083 | 0.167 |

SemIf ranks **6th of 10**. It is tied with `recency` at b0.05, marginally above
`bm25_lexical`, and **below a three-line structural rule** that only checks
coverage plus test-filename matching. Against XGBoost it is not close: 0.317 vs
0.733 at b0.05, a factor of 2.3.

This is the feasibility answer. On this benchmark SemIf does not compete with
gradient-boosted trees, and it does not beat the cheapest non-learned baselines by
a meaningful margin.

Caveats before treating this as final: n=60 (wide CIs), one instruction wording,
one orientation, and the `covered` candidate set. The full 464-fault run and the
change-shuffle ablation are running to firm it up.


### Starved regime: SemIf overtakes XGBoost

Filter: changes whose killing `(file, test)` pair has almost no failure history
(`dataset.starved_mask`). This targets the data-starved deployment regime -- a huge
codebase where a file has not changed in years and a long-running test has been run
against it once or twice. Mutation testing does not reproduce that exactly, but it
is the closest available proxy, and it selects precisely the changes for which
failure-history features carry no information.

**failures <= 2** -- 43 faults, failure count mean 1.61, **sd 0.49, max 2**:

| model | b0.01 | b0.05 | b0.10 | b0.20 |
|---|---|---|---|---|
| **`semif_textonly`** | **0.302** | **0.442** | **0.535** | **0.628** |
| `xgboost_struct_lex` | 0.116 | 0.256 | 0.395 | 0.558 |
| `xgboost_static` | 0.070 | 0.233 | 0.349 | 0.442 |
| `bm25_lexical` | 0.209 | 0.233 | 0.302 | 0.419 |
| `xgboost_struct` | 0.047 | 0.140 | 0.209 | 0.442 |
| `failure_rate` | 0.023 | 0.070 | 0.116 | 0.233 |
| `recency` | 0.000 | 0.000 | 0.070 | 0.163 |
| `coverage` | 0.000 | 0.000 | 0.000 | 0.093 |

Paired bootstrap, SemIf as reference (2000 resamples):

| budget | k | SemIf recall [95% CI] | vs `struct_lex` | vs `static` | vs `bm25` |
|---|---|---|---|---|---|
| 0.01 | 3 | 0.302 [0.163, 0.442] | +0.186 p=0.019 | +0.233 p=0.006 | +0.093 p=0.201 |
| 0.05 | 10 | 0.442 [0.302, 0.605] | +0.186 p=0.045 | +0.209 p=0.050 | +0.209 p=0.008 |
| 0.20 | 39 | 0.628 [0.488, 0.767] | +0.070 p=0.447 | +0.186 p=0.099 | +0.209 p=0.024 |

**The effect is monotone in starvation**, which is stronger evidence than any single
cell:

| regime | faults | SemIf | best XGBoost | winner |
|---|---|---|---|---|
| full set | 464 | 0.306 | 0.631 | XGBoost 2.1x |
| failures <= 5 | 141 | 0.418 | 0.489 | XGBoost 1.2x |
| **failures <= 2** | **43** | **0.442** | **0.256** | **SemIf 1.7x** |

Mechanism, and it is coherent: `failure_rate` and `recency` collapse to 0.023-0.070
(the history features are genuinely dead, as the filter intends), while SemIf's
*relative* recall **rises** from 0.306 to 0.442 as history is removed and XGBoost's
**falls** from 0.631 to 0.256. SemIf is the model that does not depend on history.

### Mirror arm: the degradation was a position artifact

The fairness arm serializes all 15 structured features (plus changed file, changed
function, test file, test name) into `<Instruct>`. Prompt grows 141 -> 343 tokens on
the starved subset. 78,371 pairs over the full held-out set; 74 min, 0 HSA errors.

**Correction.** An earlier version of this document concluded that "supplying the
features does not rescue SemIf; it actively degrades it". **That conclusion was
wrong.** The degradation is caused by *where* the block was placed, not by its
contents. Four diagnostic arms on the starved subset isolate this.

#### Controls on the starved subset (141 changes, 141 faults)

| arm | what it varies | tokens added |
|---|---|---|
| `textonly` | reference | 0 |
| `informative` | only the 8 features that vary across candidates | +143 |
| `placebo` | same field names/length, every value `n/a` | +183 |
| `full` | all 15 features (original mirror) | +202 |
| `shuffled` | real values, decorrelated from the candidate | +204 |
| `after_document` | all 15 features, placed **after** `<Document>` | +202 |

Recall, and paired difference vs `textonly` at b0.05:

| arm | b0.01 | b0.05 | b0.10 | b0.20 | vs textonly (b0.05) |
|---|---|---|---|---|---|
| `xgboost_struct_lex` | 0.298 | 0.489 | 0.624 | 0.730 | +0.071 n.s. |
| **`ctl_after_doc`** | **0.333** | **0.454** | 0.539 | 0.617 | **+0.035 n.s.** |
| `semif_textonly` | 0.326 | 0.418 | 0.539 | 0.674 | reference |
| `bm25_lexical` | 0.284 | 0.355 | 0.397 | 0.553 | -0.064 n.s. |
| `ctl_informative` | 0.220 | 0.255 | 0.305 | 0.362 | -0.163 **SIG** |
| `ctl_placebo` | 0.191 | 0.213 | 0.277 | 0.411 | -0.206 **SIG** |
| `ctl_shuffled` | 0.177 | 0.213 | 0.234 | 0.348 | -0.206 **SIG** |
| `semif_full` | 0.177 | 0.206 | 0.234 | 0.319 | -0.213 **SIG** |

**Reading:**

1. **Information content is irrelevant.** `placebo` (every value `n/a`, zero
   information) degrades by -0.206, statistically the same as `full` (real values)
   at -0.213 and `shuffled` (real but decorrelated values) at -0.206. A block with
   no information at all hurts exactly as much as the informative one.
2. **Position is the mechanism.** `after_document` carries the identical 15 features
   at the identical length but is placed *after* the Document, and it is
   indistinguishable from text-only (+0.035, p=0.32). It is also the best SemIf
   configuration at b0.01 (0.333), above text-only (0.326) and XGBoost (0.298).
3. **Length explains the residual ordering** among the pre-content arms:
   +143 (0.255) > +183 (0.213) ~ +202 (0.206). More tokens before the content is
   monotonically worse.

The mechanism is what you would expect from a reranker that reads the final
position: pushing the Query/Document further from the end dilutes the signal it is
trained to read. Inserting ~200 tokens of *anything* in `<Instruct>` does this.

#### Corrected conclusion on the fairness question

Giving SemIf the structured features is **neutral, not harmful**, provided the block
goes after the content:

| regime | b0.01 | b0.05 | b0.20 |
|---|---|---|---|
| `textonly` | 0.302 | 0.442 | 0.628 |
| `ctl_after_doc` | 0.349 | 0.442 | 0.581 |

And on the starved `failures <= 2` subset, paired against XGBoost:

| arm | b=0.01 | b=0.05 |
|---|---|---|
| `ctl_after_doc` | **+0.233** [+0.070, +0.395] p=**0.002** | **+0.186** [+0.023, +0.372] p=**0.042** |
| `semif_textonly` | +0.186 [+0.047, +0.349] p=0.019 | +0.186 [+0.023, +0.372] p=0.045 |
| `semif_full` | +0.023 n.s. | -0.047 n.s. |

So the correctly-placed fairness arm is the strongest SemIf variant and beats
XGBoost with a better p-value than text-only. But it does not beat text-only, so the
features are not what drives the advantage -- the text reading is.

**Still to do:** the full held-out mirror was scored with the wrong placement, so the
"mirror is worse everywhere" claim on the full 464-fault set is unverified for
correct placement. Re-running `after_document` placement over all 530 held-out
changes costs ~74 min and should be done before citing the full-set comparison.


### The full regime cannot be tested with adequate power

The regime described is "file unchanged for years **and** test run once or twice".
Applying the run cap as well as the failure cap collapses the sample:

| filter | held-out faults | runs mean | runs max |
|---|---|---|---|
| failures<=2, runs=any | 43 | 54.4 | 392 |
| failures<=2, runs<=40 | 27 | 17.2 | 40 |
| failures<=2, runs<=20 | 15 | 10.1 | 20 |
| failures<=2, runs<=10 | 8 | 5.0 | 10 |

So the failure cap alone is the practical proxy; adding the run cap leaves too few
faults for a meaningful interval. This is a limitation of the benchmark, not of the
filter.

### How strong is this evidence?

Moderate, not decisive. Stated plainly:

1. **The winning margins are marginal.** p=0.045 and p=0.050 at b0.05 would not
   survive a multiple-comparison correction; the sweep is 2 thresholds x 3 budgets
   x 4 baselines = 24 tests, so ~1 false positive is expected at alpha=0.05. The
   b0.01 result (p=0.019 vs `struct_lex`, p=0.006 vs `static`) is the most robust.
2. **The threshold was chosen after seeing the data.** Thresholds 1/2/3/5/8 were
   explored, then 2 and 5 reported. The monotone trend mitigates this but does not
   remove it; a pre-registered threshold would be stronger.
3. **n=43.** SemIf's b0.05 interval spans [0.302, 0.605].
4. **The advantage disappears at larger budgets** (b0.20, p=0.447). Once both models
   have enough slots, they converge. The gain is a small-budget ranking effect.
5. **At failures <= 5 it is a draw, not a win** -- SemIf is behind `struct_lex` at
   b0.05 (-0.071, p=0.182) and b0.20 (-0.057, p=0.264), both non-significant.
6. **Only failure history was filtered.** `max_runs` (the "run once or twice" half of
   the regime) was not applied; `runs` still average 54 in the <=2 arm.
7. **`covered` candidate mask** makes `coverage` 0.000 by construction, so this arm
   measures ordering *within* the covered set, not coverage selection.
8. Single instruction wording, single orientation, one SUT, one revision.

### What would strengthen it

- Pre-register the starvation threshold, or report the full threshold sweep.
- Add `max_runs` to the filter to match the "run once or twice" condition.
- Re-score the starved changes against the full 1187-test suite so `coverage` is not
  degenerate (~35 min).
- Test several instruction wordings; the mirror arm predicts features will *not* help
  here, since they are uninformative by construction.


### Cost

Superseded numbers kept for the record. The original estimate assumed the repo's
batch-1 figure of 1.86 decisions/s; measured batched throughput is 17-24 pairs/s,
so real cost is far lower.

| scope | pairs | at 1.86/s (assumed) | measured |
|---|---|---|---|
| Full grid (all changes) | 410k | 61 h | ~5 h |
| **Held-out only** | 78k | 12 h | **54 min** (text-only), ~75 min (mirror) |
| Held-out + change-shuffle | 157k | 24 h | ~2 h |

`rts/semif.py`'s `--max-changes` still slices from the start of the history and so
includes training changes; `rts/semif_runner.py`'s `--heldout` is the mode actually
used, and SemIf being zero-shot means the training window is never needed.


### Correction: XGBoost feature importances

An earlier version of this document stated that `covers_function`,
`n_covering_tests`, and `coverage_rank_prior` account for 94.6% of importance.
**That figure was an artifact.** The original selector trained on *all* 1187 tests
per change, including non-candidates. `covers_function` then does real work -- it
separates candidates from non-candidates -- but that job does not exist at
evaluation time, where every test is already a candidate.

Training on the candidate pairs actually being ranked (which the bundle ladder does,
and which the original selector did not):

| feature | train on all pairs (artifact) | train on candidate pairs (correct) |
|---|---|---|
| `covers_function` | 0.7143 | *not in the top 8* |
| `coverage_rank_prior` | 0.1012 | 0.4206 |
| `test_last_failure_age` | 0.0163 | 0.1261 |
| `test_duration` | 0.0175 | 0.0648 |
| `n_covering_tests` | 0.1306 | 0.0577 |
| `test_failure_rate_cum` | 0.0026 | 0.0563 |
| `module_name_in_test_file` | 0.0041 | 0.0495 |
| recall @0.05 | 0.631 | 0.616 |

The corrected picture is that **no single feature dominates**. Importance is spread
across six or seven individually weak cues. Note also that `coverage_rank_prior` and
`n_covering_tests` are *constant within a change*, so they cannot order tests
directly; they earn their place by gating the others ("if the coverage set is small,
trust the filename match").

**Importances are also unstable.** In the bundle ladder, which lacks
`coverage_rank_prior`, `name_match_any` becomes the top feature at 0.17-0.22 instead
of 0.05. Same data, different feature set, different "most important" feature -- the
features are partly substitutable. Treat importances as descriptive only.

## Change complexity

Bundling one killed mutant with **survived** distractors broadens the change while
keeping the label exact (survived mutants have no killing tests, so the union kill
set is unchanged). The candidate pool is held fixed at the signal's covered set, so
only the change *description* varies. See `rts/bundles.py`.

**200 held-out bundles, recall @ budget 0.05:**

| change | files | SemIf | BM25 | XGBoost | structural rule | random |
|---|---|---|---|---|---|---|
| 1 mutation (baseline) | 1 | 0.305 | 0.235 | 0.550 | 0.280 | 0.090 |
| 6 mutations | 1 | 0.235 | 0.145 | 0.560 | 0.280 | 0.090 |
| 6 mutations | 3.8 | **0.190** | 0.160 | 0.525 | 0.300 | 0.090 |

Paired bootstrap against the baseline:

| model | 6 mutations, 1 file | 6 mutations, 3.8 files |
|---|---|---|
| SemIf | -0.070 [-0.125, -0.015] p=0.013 | **-0.115** [-0.175, -0.060] p<0.0001 |
| BM25 | -0.090 [-0.140, -0.040] p<0.0001 | -0.075 [-0.120, -0.030] p=0.001 |
| XGBoost | +0.010 n.s. | -0.025 n.s. |
| structural rule | 0.000 n.s. | +0.020 n.s. |

**Increasing change complexity hurts text models and leaves structural models
untouched.** XGBoost is flat even with the change spanning 3.8 files. The mechanism
is query dilution: the bundled diff concatenates several unrelated edits, so the
distractor terms wash out the signal terms. This is the same effect measured in the
placement controls, where ~200 tokens of *uninformative* text cost SemIf 0.21
recall.

XGBoost survives because dilution is replaced by a new informative feature,
`n_mutations_covered` (how much of the bundle a test covers), which is why its top
features shift to `name_match_any` and `test_duration` rather than collapsing.

**The SemIf-vs-BM25 gap does not widen.** This was the pre-registered falsifier for
the paraphrase hypothesis -- the one thing a reranker does that bag-of-words cannot:

| | SemIf vs BM25 |
|---|---|
| 1 mutation | +0.070 [+0.010, +0.135] p=0.023 |
| 6 mutations, 1 file | +0.090 [+0.035, +0.145] p<0.0001 |
| 6 mutations, 3.8 files | +0.030 [-0.025, +0.085] p=0.366 n.s. |

The gap is flat within noise at one file and *collapses to non-significance* at
3.8 files. So complexity buys SemIf nothing; at cross-file breadth it costs SemIf
its advantage over BM25 entirely.

**Rung 4 (killed distractors) confirms the killer-count trap.** With 5 *killed*
distractors, `random` jumps from 0.090 to 0.360 and XGBoost to 0.975. More killing
tests makes RTS easier, not harder -- which is why survived distractors are the
right choice for this manipulation and killed ones are a separate control.

**Caveat that limits the claim.** Distractors are picked uniformly at random, so
they are semantically *unrelated* to the signal. That is the worst case for a text
model and a neutral case for coverage. Real complex commits are usually *coherent*
-- a refactor touches one concept across files -- so this manipulation may be
unfairly adversarial to text models. A coherent-bundle variant (distractors chosen
by relatedness) has not been run and is the obvious next step.

## Iteration cost and data sizing

Measured, not estimated. All figures come from the runs recorded above.

### Where the time goes

SemIf scores are **cached**, so scoring is a one-time cost per arm, not a
per-iteration cost. Everything downstream is CPU-only:

| step | now | after the training-set fix |
|---|---|---|
| dataset build | ~10 s | ~10 s |
| structured features | 1.2 s | 1.2 s |
| BM25 | 0.4 s | 0.4 s |
| XGBoost fits (7 variants) | ~140 s | ~21 s |
| evaluation + bootstrap | ~10 s | ~10 s |
| **total** | **~2.7 min** | **~45 s** |

**XGBoost is ~80% of iteration cost, and most of it is wasted.** Training uses all
1187 tests x 2121 changes = 2.5M rows, but evaluation only ever ranks within the
covered mask (~155 candidates). The model therefore trains on ~1032 candidates per
change that it is never asked to rank. Restricting training to covered candidates
gives 329k rows instead of 2.5M (7.7x fewer) and should cut each fit from ~20 s to
~3 s. This is a distribution-matching fix rather than a shortcut: training and
evaluation should cover the same candidate set. **Not yet implemented.**

### Recommended scoring sizes

At 20 pairs/s and 154.7 covered candidates per change:

| tier | changes | pairs | time/arm | detects effects >= |
|---|---|---|---|---|
| smoke | 5-10 | ~1.5k | seconds | nothing |
| arm development | 43 | 6.7k | 6 min | 0.35 |
| **sweet spot** | **141** | **21.8k** | **18 min** | **0.20** |
| decision grade | 250 | 38.7k | 32 min | 0.15 |
| full | 464 | 71.9k | 60 min | 0.08 |

**Use 141 changes** -- the `failures <= 5` starved population. It is a superset of
`failures <= 2` (43 faults), so a single scoring run serves both starved thresholds
at 30% of the full cost. `--heldout` currently scores all 464; scoring a subset
needs a new flag.

### What n is needed for which comparison

Effect sizes are the measured paired differences; required n is where the paired CI
half-width sits comfortably below half the effect.

| comparison | effect | needed n |
|---|---|---|
| SemIf vs XGBoost, full set | 0.325 | ~50 |
| mirror vs text-only, starved | 0.233 | ~100 |
| SemIf vs XGBoost, starved <=2 | 0.186 | ~141 |
| SemIf vs XGBoost, starved <=5 | 0.071 | ~500 |
| SemIf vs BM25, full | 0.037 | ~2000 |

Two consequences:

- **141 is the smallest n that can test the main hypothesis**, since the starved
  effect is 0.186 and the observed paired CI half-width at n=141 is +-0.10.
- **The SemIf-vs-BM25 difference (0.037) is undetectable at any feasible n on this
  benchmark.** That comparison will always return "not significant" -- a power
  limit, not evidence of equivalence. Do not spend compute chasing it.

### Constraints

- **The starved populations are fixed.** 43 and 141 are the entire populations, so
  they cannot be subsampled without losing power. Score all of them or skip the arm.
- **`--pilot N` is biased.** It takes the first N held-out faults by index, which is
  a temporal slice (the earliest held-out changes), not a random sample.
- **A prompt change invalidates the cache.** The instruction sweep is the one lever
  that cannot be amortized: each wording costs a full scoring run. At 18 min per
  wording on the 141-change set, a 5-wording sweep is ~90 min -- a further argument
  for 141 over 464.

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

