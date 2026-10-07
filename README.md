# A benchmarking harness for regression test selection

A harness for measuring regression-test-selection (RTS) models. It defines what a *dataset* is,
what a *selection model* is, how a *sweep* is declared, and how results are recorded — and it is
generic: nothing in `rts/` knows about a particular study, dataset or model.

## What it guarantees

- **Leakage-free, protocol-correct evaluation.** The split is a value, the evaluation window is
  enforced, and a metric can only be averaged over rows the split held out. Per-change budgets
  mean a constant-selection strategy cannot score well.
- **A sweep is a value.** An experiment is a Python module that combines reusable pieces; running
  it measures every design point. The declaration contains no logic.
- **Unmeasurable is not zero.** A quantity that cannot be defined stays `Undefined` and is
  *reported*, never silently dropped or coerced to `0` or an empty average.
- **One meaning per statistic.** Every derived quantity is a harness function over the dataset
  contract, so an identical column means an identical thing across datasets.
- **Datasets compose.** Pooling, partitioning and derivation are iteration over one contract.

## Layout

```
rts/                     the harness (generic; no study)
  config.py                paths and evaluation defaults
  data/                    the dataset half
    contract.py              primitives, declarations, and Undefined
    accessors.py             derived quantities over the contract
    splits.py                the train/test split, and the window guard
    subsets.py               named evaluation subsets, and the registry mechanism
    composition.py           namespacing, pooling, derived datasets
  features/                the declared feature blocks (structured + text)
  model/rankers.py         the ranker interface, the context, and cached/produced scores
  evaluate.py              the metric sweep and the paired bootstrap
  reporting.py             describing a dataset, and auditing what a consumer should know
  experiment/              the layer: declaration, run, report
examples/                one study's plugins, and a declarative example sweep
tests/                   the kernel test suite
```

`examples/` is **a** study, not the harness: it implements the interfaces (concrete datasets,
rankers and feature blocks) and shows how a sweep is declared. The dependency runs one way — the
kernel never imports `examples/`.

## Using it

Run the declarative example — no checkout, no GPU, no artifacts:

```
python -m examples.example
```

Write a study by combining pieces into an `rts.experiment.Experiment` value and calling `run`.
`examples/example.py` is the whole thing: a dataset, a feature block, two rankers, a subset and a
split, combined with no loops, cache paths or closures.

## Development

```
python -m pytest tests/ -q
python -m ruff check rts/ tests/ examples/ conftest.py
```

## Documentation

Each module's docstring states what it is and why it is shaped that way; `examples/README.md`
explains the plugin layout. There is no separate design-doc set yet — the code is the record.
