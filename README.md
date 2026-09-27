# Discriminative transformer RTS

Does a semantic reranker beat cheap structure for regression test selection (RTS)? This repo is
the feasibility study that answers it, on a mutation-testing benchmark and then on real data —
and the harness that makes the measurements reproducible.

**Bottom line: on this benchmark SemIf loses in every regime where coverage is available, and wins
only once it is removed.** With the coverage-based candidate mask, a coverage + BM25 tree reaches
0.700 recall at budget 0.05 against SemIf's 0.306 (`artifacts/results_covered.json`). On the
real-label corpus (BugsInPy, which has no coverage at all) SemIf ties BM25 rather than beating it.
The full numbers, the four text-side levers that were tried and failed, and what would change the
verdict are in [`docs/handoff.md`](docs/handoff.md).

## Layout

```
rts/              the harness
  config.py         paths, pinned choices, seeds
  data/             the dataset half: the contract, and where a dataset comes from
    contract.py       primitives, declarations, and the values they speak in
    accessors.py      derived quantities over the contract, and the material catalogue
    splits.py         the train/test split, and the evaluation-window boundary
    populations.py    named subsets of the evaluation window
    composition.py    namespacing, pooling, derived datasets
    datasets.py       the concrete datasets
    sources.py        the raw material for one SUT or revision
    mutmut.py         mutmut's raw artifacts, and the changes they become
    test_source.py    test-function source text by pytest node id
    reporting.py      describe, audit, recurrence
  features/         the declared feature blocks (structured, bundle, text)
  model/            the model half: what turns a change and a test into a score
    selectors.py      selectors: rules, BM25, XGBoost, cached and produced scores
    semif.py          the pinned reranker adapter, and its cache format
    semif_runner.py   the pairwise scorer (what spends GPU time)
    direct_runner.py  the direct-mode scorer
    embed.py          the code-embedding baseline
  evaluate.py       the metric sweep and the paired bootstrap
  experiment/       the layer: axes, elements, cells, and a run
    declaration.py    what an experiment is, as a value (roles, axes, elements, cells)
    report.py         what a run records (the measured cell, and the report)
    run.py            measuring it: requirements, the cell loop, the comparison
  studies/          the study's choices, as values (axes, arms, readings)
  render/           what runs an arm or reads an artifact
    pipeline.py       renders results_{full,covered}.json
    ladder.py         renders ladder.json (the traceability-loss ladder)
    variations.py     renders variations.json (the six variation sections)
    bugsinpy.py       renders bugsinpy_results.json (real labels, no coverage)
    panels.py         the sparsity panels (an artifact reader, not a sweep)
    figures.py        the variation figures (an artifact reader)
  bundles.py        the change-complexity ladder -- the last driver outside rts.render
scripts/          entry points, probes and one-off builders (see scripts/README.md)
tests/            the unit and semantics suite
artifacts/        the recorded results — the *specification*, not a cache
docs/             the design and status records (see below)
sut/              external checkouts (gitignored): marshmallow, SemIf, BugsInPy
```

## Reproducing

The study's numbers are recorded in `artifacts/`, and the migrated arms are checked against them
by regenerating and comparing **leaf by leaf**. `check_fast.py` is the one to run per commit:
about 50 s, and it compares the headline arm's non-fitting selectors and the BugsInPy arm
against the recorded artifacts, through the real drivers. `verify_experiment_layer.py` is the
full gate -- every arm, about 30 min, and the acceptance criterion.

```
python scripts/check_fast.py                          # ~50 s: static, unit, cheap numbers, real data
python scripts/check_fast.py --quick                  # ~30 s: static and unit only
python scripts/verify_experiment_layer.py            # every migrated arm (~30 min)
python scripts/verify_experiment_layer.py bugsinpy    # one arm (~6 s)
python -m pytest tests/ -q                            # the unit suite
python -m ruff check rts/ tests/ scripts/ conftest.py  # the lint config (pip install ruff)
```

A recorded artifact is the specification: if a change moves a number, either the change is wrong
or the movement is a finding to argue and record — never absorbed.

Run a sweep, or one arm of it:

```
python -m rts.render.pipeline                                # the headline arm
python -m rts.studies ladder.mutmut --tiers cpu        # a declared arm, as a value
python -m rts.studies semif.produce --tiers gpu        # produce a score cache as a cell
```

Score caches are resumable JSONL, so a re-run continues rather than restarts. Only one 4B model
fits in 17 GB, so the SemIf arms run sequentially: `scripts/run_variation_arms.sh` queues them.

## Documentation

| record | authoritative for |
|---|---|
| [`docs/plan.md`](docs/plan.md) | the study's design: scope, evaluation protocol, what is undecided, next steps |
| [`docs/plan_next_steps.md`](docs/plan_next_steps.md) | the execution plan for the two gaps, with gates and costs |
| [`docs/implementation.md`](docs/implementation.md) | how the study is built and **what was measured** — the results record |
| [`docs/refactor.md`](docs/refactor.md) | the dataset-contract refactor: what it fixed and why |
| [`docs/experiment.md`](docs/experiment.md) | the experiment layer's design: roles, cells, availability, cost tiers |
| [`docs/handoff.md`](docs/handoff.md) | **status**: the study's next steps (section 1) and the harness's remaining work (section 2) |

Start at `docs/handoff.md` for status, and `docs/implementation.md` for what the numbers are. When
two records overlap, the more specific one is authoritative: `experiment.md` for the layer's
design, `handoff.md` for its status.
