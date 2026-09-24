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

## Remaining risks

- **Test-selection assumption.** mutmut's selection (coverage plus `max_stack_depth`) assumes tests outside the associated set cannot fail. Spot-check a handful of mutants by running the full suite to confirm the association is not dropping real killers.
- **Pretraining contamination.** Cannot be ruled out for a well-known project; state as a limitation rather than trying to fix it in a probe.
