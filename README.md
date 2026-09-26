# Discriminative transformer RTS

Does a semantic reranker beat cheap structure for regression test selection (RTS)? This repo is
the feasibility study that answers it, on a mutation-testing benchmark and then on real data —
and the harness that makes the measurements reproducible.

**Bottom line: on this benchmark SemIf loses in every regime where coverage is available, and wins
only once it is removed.** With the full feature set, a coverage + BM25 tree reaches 0.700 recall
at budget 0.05 against SemIf's 0.306. On the real-label corpus (BugsInPy, no coverage at all)
SemIf ties BM25 rather than beating it. The full numbers, the four text-side levers that were
tried and failed, and what would change the verdict are in
[`docs/handoff.md`](docs/handoff.md).

## Layout

```
rts/              the harness
  contract.py       the dataset contract: primitives and declarations
  accessors.py      derived quantities over the contract, and the material catalogue
  splits.py         the train/test split, and the evaluation-window boundary
  features/         the declared feature blocks (structured, bundle, text)
  models.py         selectors: rules, BM25, XGBoost, cached scores, per-change scope
  evaluate.py       the metric sweep and the paired bootstrap
  experiment.py     the layer: axes, elements, cells, and a run
  studies/          the study's choices, as values (axes, arms, readings)
  pipeline.py       renders results_{full,covered}.json
  ladder.py         renders ladder.json (the traceability-loss ladder)
  variations.py     renders variations.json (the six variation sections)
  bugsinpy.py       renders bugsinpy_results.json (real labels, no coverage)
  bundles.py        the change-complexity ladder
  panels.py         the sparsity panels (an artifact reader, not a sweep)
  figures.py        the variation figures (an artifact reader)
scripts/          entry points, probes and one-off builders (see scripts/README.md)
tests/            the unit and semantics suite
artifacts/        the recorded results — the *specification*, not a cache
docs/             the design and status records (see below)
sut/              external checkouts (gitignored): marshmallow, SemIf, BugsInPy
```

## Reproducing

The study's numbers are recorded in `artifacts/`, and the migrated arms are checked against them
by regenerating and comparing **leaf by leaf**:

```
python scripts/verify_experiment_layer.py            # every migrated arm
python scripts/verify_experiment_layer.py bugsinpy    # one arm
python -m pytest tests/ -q                            # the unit suite
```

A recorded artifact is the specification: if a change moves a number, either the change is wrong
or the movement is a finding to argue and record — never absorbed.

Run a sweep, or one arm of it:

```
python -m rts.pipeline                                # the headline arm
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
