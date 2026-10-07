# examples/

Concrete plugins for one study, and a declarative example that combines them.

Nothing here is part of the harness. A module in this package *implements* one of the kernel's
interfaces:

| file | what it implements |
|---|---|
| `datasets.py`, `sources.py`, `mutmut.py`, `test_source.py` | `rts.data.contract.Dataset` over a mutation-testing checkout and a real-bug corpus |
| `rankers.py`, `semif.py`, `semif_runner.py`, `direct_runner.py`, `embed.py` | `rts.model.rankers.Ranker` (baselines, trees, and cached/produced transformer scores) |
| `bundle_features.py` | a `rts.features.block.FeatureBlock` (the change-complexity ladder) |
| `subsets.py` | `rts.data.subsets.Subset` values (the study's deployment proxies) |
| `fixture.py` | a tiny in-memory `Dataset`, for tests and the example |
| `config.py` | the study's pins: checkpoints, the checkout, artifact paths |

`example.py` is the point of the package: a complete sweep written as **data**. It combines
reusable pieces into an `rts.experiment.Experiment` value and calls `run` — no loops, no cache
paths, no closures. Run it with:

```
python -m examples.example
```

## The one-way rule

This package imports `rts`. `rts` never imports this package. That is what keeps the harness
generic and separable from any single study.
