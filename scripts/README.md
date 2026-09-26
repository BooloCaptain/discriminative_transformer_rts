# Scripts

Two entry points live here; everything else is grouped by what it was for. Nothing in this
directory is imported by `rts/` -- a script is a *caller* of the harness, never part of it.

## Entry points

| script | what it does |
|---|---|
| `verify_experiment_layer.py` | **the gate.** Runs each migrated arm and compares the artifact it renders against the recorded one, leaf by leaf. Exits non-zero on a missing leaf, a differing leaf, or an unexpected leaf that is not on the documented addition list. |
| `run_variation_arms.sh` | queues the SemIf arms. Only one 4B model fits in 17 GB, so they run sequentially rather than in parallel. |

## `probes/` -- reconnaissance, run once each

Historical: these produced the evidence that decided a direction, and they are kept because
their findings are cited. They are not part of the reproduction recipe.

| script | question it answered |
|---|---|
| `probe_full_suite.py` | what happens if every mutant is run against the *whole* suite instead of mutmut's median of 5 tests (Gap 1) |
| `summarise_full_suite_probe.py` | turns the probe's output into the summary the corrected labels are built from |
| `micropython_bridge_probe.py` | whether the lexical bridge survives a real cross-boundary suite (it does) |

## `data/` -- one-off builders

Also historical, and each is idempotent: they rebuild an artifact from a checkout under `sut/`,
which is gitignored, so they only run on a machine that has that checkout.

| script | builds |
|---|---|
| `build_bugsinpy_dataset.py` | `artifacts/bugsinpy/<project>.json`, one per project |
| `emit_full_suite_outcomes.py` | the per-test outcome records from `artifacts/full_suite_labels.json` |
| `complete_semif_full_cache.py` | extends the `failures <= 2` cache to `failures <= 5`, scoring only the new changes |

## `_workspace.py`

Locates the workspace root by searching upward for a marker (`rts/__init__.py` next to
`conftest.py`) instead of counting `parent`s. A script one directory deeper would otherwise
compute a path to the wrong directory, and fail later as a missing dataset rather than at import.
