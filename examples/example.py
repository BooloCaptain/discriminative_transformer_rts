"""A study, as a pure declaration: modular pieces combined, with no logic of its own.

This is the template the harness exists to make possible. Every value is a reusable piece -- a
dataset, a feature block, a ranker, a subset, a split -- and the experiment is the *combination*.
There is no loop, no cache path, no closure and no branch in the declaration: the only executable
line is the call to :func:`rts.experiment.run`.

    python -m examples.example

The fixture dataset keeps the example runnable with no checkout, no GPU and no artifacts.
"""

from __future__ import annotations

from examples.fixture import StubDataset
from examples.rankers import CoverageRanker
from rts import features
from rts.data import subsets
from rts.experiment import (
    FACTOR_BUDGET,
    FACTOR_DATASET,
    FACTOR_FEATURES,
    FACTOR_MODEL,
    FACTOR_SPLIT,
    FACTOR_SUBSET,
    Contrast,
    Experiment,
    Factor,
    budget_level,
    constant,
    run,
    split_level,
)
from rts.model.rankers import RandomRanker

#: The experiment, as a value. Building it runs nothing.
EXAMPLE = Experiment(
    name="example.stub",
    datasets=Factor(FACTOR_DATASET, (constant("stub", StubDataset()),)),
    features=Factor(FACTOR_FEATURES, (constant("structured", features.STRUCTURED),)),
    models=Factor(
        FACTOR_MODEL,
        (
            constant("random", RandomRanker()),
            constant("coverage", CoverageRanker()),
        ),
    ),
    subsets=Factor(FACTOR_SUBSET, (constant("detectable", subsets.DETECTABLE),)),
    splits=Factor(FACTOR_SPLIT, (split_level(train_fraction=0.5),)),
    budgets=Factor(FACTOR_BUDGET, (budget_level(0.05), budget_level(0.2))),
    contrasts=(Contrast(FACTOR_MODEL, "random", 0.05),),
    note="a minimal declarative sweep over the fixture dataset",
)


def main() -> None:
    report = run(EXAMPLE, save=False, verbose=False)
    for result in report.design_points:
        for row in result.results:
            print(f"{result.ranker:>10}  b{row['budget']:.2f}  recall={row['recall']:.3f}")


if __name__ == "__main__":
    main()
