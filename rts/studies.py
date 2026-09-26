"""The study's experiments, expressed as values: axes by role, and the named arms.

This is the config module the experiment layer was built for. It holds the study's *choices*
-- which datasets, which rungs, which selectors, which populations, which budgets -- and
nothing about how a sweep is executed.

**Axis builders are functions, not module constants**, wherever the elements hold state. An
:class:`~rts.experiment.Experiment` is immutable and could be a constant, but its axes hold
selector instances that train on use, so a fresh axis per arm keeps two runs from sharing one
model object.

**Two arms here are verification vehicles.** ``study_arm()`` reproduces
``artifacts/results_full.json`` and ``ladder_arm()`` reproduces ``artifacts/ladder.json``
(``experiment.md`` §11). Until they do, the hand-written drivers stay in place as the reference
implementation -- which is also why the selector lists and rung definitions below are written
out here rather than read back out of ``ladder.py``: they are study choices and belong with the
study's config, and the duplication disappears when the drivers do.

Usage::

    python -m rts.studies study
    python -m rts.studies ladder.mutmut --tiers cpu cache
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from . import bundles, config, datasets, features, ladder, models, populations, splits
from .contract import Unmeasured
from .experiment import (
    ROLE_DATASET,
    ROLE_FEATURES,
    ROLE_MODEL,
    ROLE_POPULATION,
    ROLE_SPLIT,
    Axis,
    Comparison,
    Element,
    Experiment,
    Knobs,
    RunReport,
    constant,
    run,
)

__all__ = [
    "ARMS",
    "dataset_axis",
    "ladder_arm",
    "ladder_feature_axis",
    "ladder_model_axis",
    "ladder_population_axis",
    "ladder_selectors",
    "main",
    "model_axis",
    "population_axis",
    "semif_margins",
    "split_axis",
    "structured_feature_axis",
    "study_arm",
]


# --- datasets ---------------------------------------------------------------


def _marshmallow(labels: str) -> Element:
    return Element(
        f"marshmallow_{labels}",
        lambda b, labels=labels: datasets.marshmallow(labels=labels, order_seed=b.knobs.seed),
        note=f"the mutation-testing dataset, {labels} labels",
    )


def _bundle_dataset(base_labels: str, rung: int, pool: str) -> Element:
    """A dataset derived from another dataset -- the pattern that needs no special support.

    ``make`` is arbitrary Python, so a derived dataset is just an element whose builder
    constructs (and here shares) its base.
    """

    def make(binding, rung=rung, pool=pool, base_labels=base_labels):
        base = datasets.marshmallow(labels=base_labels, order_seed=binding.knobs.seed)
        return datasets.bundles(base, rung, seed=binding.knobs.seed, pool=pool)

    kind = bundles.RUNGS[rung][2]
    return Element(
        f"bundles_r{rung}_{pool}_{base_labels}",
        make,
        note=f"rung {rung} ({kind} distractors), pool={pool}",
    )


def dataset_axis(
    label_sources: Sequence[str] = ("mutmut", "full"),
    *,
    bundle_rungs: Sequence[int] = (),
    pool: str = "signal",
) -> Axis:
    """One element per label source, plus one per bundle rung when asked for.

    A label source is a dataset, not a knob, because a source is what the labels come from:
    the refactor made it a constructor argument precisely so two label sources can coexist in
    one process, and an axis is how that gets expressed.
    """
    elements = [_marshmallow(labels) for labels in label_sources]
    elements.extend(
        _bundle_dataset(label_sources[0], rung, pool) for rung in bundle_rungs
    )
    return Axis(ROLE_DATASET, tuple(elements), note="one dataset per label source")


# --- feature sets -----------------------------------------------------------


def structured_feature_axis() -> Axis:
    return Axis(
        ROLE_FEATURES,
        (constant("structured", features.STRUCTURED, tier="cpu", note="the 15 declared columns"),),
        note="the one block the study's headline arm uses",
    )


def ladder_feature_axis() -> Axis:
    """The ladder's rungs, as blocks with families withheld.

    The rung *definition* is a study choice and stays in ``rts.ladder``; what the layer does is
    turn each one into an element. Withholding a family goes through the block's own
    ``without_families``, so a rung takes the same unmeasured path a genuinely absent
    capability takes and a typo raises instead of quietly ablating nothing.
    """
    elements = []
    for name, removed in ladder.RUNGS:
        block = (
            features.STRUCTURED
            if not removed
            else features.STRUCTURED.without_families(*removed)
        )
        elements.append(
            constant(
                name,
                block,
                tier="cpu",
                note=f"withheld families: {', '.join(removed) if removed else 'none'}",
            )
        )
    return Axis(ROLE_FEATURES, tuple(elements), note="the traceability ladder's rungs")


# --- models -----------------------------------------------------------------


def _model_element(selector: models.Selector, tier: str = "cpu", note: str = "") -> Element:
    """An element for a selector.

    ``tier`` is the caller's, not the selector's: it says whether *measuring* this element
    needs a GPU, which no selector in the study does -- SemIf reads a precomputed cache, which
    is why it declares an artifact requirement instead.
    """
    return Element(
        selector.name,
        lambda _binding, selector=selector: selector,
        tier=tier,
        note=note or type(selector).__name__,
    )


def model_axis(selectors: Sequence[models.Selector] | None = None) -> Axis:
    chosen = list(selectors) if selectors is not None else models.default_selectors()
    return Axis(
        ROLE_MODEL,
        tuple(_model_element(s) for s in chosen),
        note="the study's selector set",
    )


def ladder_selectors() -> list[models.Selector]:
    """The ladder's selector set.

    A study choice, so it lives with the config rather than being read back out of the driver.
    The two lists must stay in step until ``ladder.py`` is deleted (``experiment.md`` §12).
    """
    return [
        models.RandomSelector(),
        models.RecencySelector(),
        models.FailureRateSelector(),
        models.CoverageSelector(),
        models.StructuralRuleSelector(),
        models.LexicalSelector(),
        models.XGBoostSelector(include_lexical=True),
        models.XGBoostSelector(include_lexical=False),
    ]


def _ladder_semif_applies(binding) -> Unmeasured | None:
    """Why the ladder's SemIf cache does not apply to this cell's population.

    The cache was built over the changes ``starved141`` selects, and ``semif.load_scores``
    fills every *uncached* pair with a sentinel rather than reporting that it has no score. So
    evaluating SemIf on the other population yields a number that looks like a measurement and
    is not. ``ladder.py`` avoided that by simply not tabulating SemIf there; declaring the
    inapplicability keeps the cell in the report, as a finding with a reason.
    """
    population = binding.factors.get(ROLE_POPULATION)
    if population == "starved141":
        return None
    return Unmeasured(
        requirement="artifact:semif_ladder_coverage",
        note=(
            "the ladder's SemIf cache covers only the starved141 changes, so it has no scores "
            f"for the {population!r} population"
        ),
    )


def ladder_model_axis(include_semif: bool = True) -> Axis:
    axis = model_axis(ladder_selectors())
    if not include_semif:
        return axis
    return axis.extend(
        Element(
            "semif_reranker",
            lambda _binding: models.SemIfSelector(scores_file=ladder.SEMIF_LADDER_CACHE),
            note="SemIf scores over the full pool, cached for the ladder's held-out subset",
            applies=_ladder_semif_applies,
        )
    )


# --- populations ------------------------------------------------------------


def population_axis(names: Sequence[str] = ("fault_bearing",)) -> Axis:
    """Named populations from the study's registry, resolved by name at declaration time."""
    return Axis(
        ROLE_POPULATION,
        tuple(
            constant(name, populations.population(name), tier="cpu", note="from populations.STUDY")
            for name in names
        ),
        note="the averaging populations a metric is reported over",
    )


def ladder_population_axis() -> Axis:
    """The ladder's two populations, built from the dataset by the ladder's own definition.

    ``build_populations`` returns an :class:`Unmeasured` when the SemIf cache is absent, which
    the layer passes through unchanged: the population is then unavailable rather than smaller,
    which is the distinction the ladder was written to preserve.
    """

    def element(name: str) -> Element:
        return Element(
            name,
            lambda binding, name=name: ladder.build_populations(binding.dataset)[name],
            tier="cpu",
            note=f"the ladder's {name} population, or why it cannot exist",
        )

    return Axis(
        ROLE_POPULATION,
        (element("heldout530"), element("starved141")),
        note="held-out change sets: one for every fault, one for the cached subset",
    )


# --- splits -----------------------------------------------------------------


def split_axis(
    train_fraction: float = config.DEFAULT_TRAIN_FRACTION,
    shuffle: bool = False,
) -> Axis:
    label = f"split{int(round(train_fraction * 100))}" + ("_shuffled" if shuffle else "")
    return Axis(
        ROLE_SPLIT,
        (
            Element(
                label,
                lambda binding: splits.make_split(
                    binding.dataset,
                    train_fraction=train_fraction,
                    shuffle=shuffle,
                    seed=binding.knobs.seed,
                ),
                tier="cpu",
                note=f"{train_fraction:.0%} train prefix" + (", shuffled" if shuffle else ""),
            ),
        ),
        note="evaluation configuration, not a property of the data",
    )


# --- the arms ---------------------------------------------------------------


def study_arm(
    label: str = "mutmut",
    *,
    candidates: str = "full",
    name: str | None = None,
) -> Experiment:
    """The study's headline arm: every selector, one dataset, one population.

    With ``candidates="full"`` and ``label="mutmut"`` this reproduces
    ``artifacts/results_full.json``.
    """
    return Experiment(
        name=name or f"study.{label}.{candidates}",
        datasets=dataset_axis((label,)),
        features=structured_feature_axis(),
        models=model_axis(),
        populations=population_axis(("fault_bearing",)),
        splits=split_axis(),
        knobs=Knobs(
            seed=config.SEED,
            budgets=config.DEFAULT_BUDGETS,
            n_bootstrap=config.DEFAULT_BOOTSTRAP,
            candidates=candidates,
        ),
        comparisons=(
            Comparison(
                ROLE_MODEL,
                "coverage",
                0.05,
                note="every selector against the coverage baseline at the study's probe budget",
            ),
        ),
        # The recorded arm explicitly enables history on an imposed order. That is an
        # override, not an option, and it produces the warning the layer carries.
        history=True,
        note="the headline arm: the full selector set, full candidate set",
    )


def ladder_arm(label: str = "mutmut", *, name: str | None = None) -> Experiment:
    """The traceability ladder: four rungs, two populations, paired against SemIf.

    Reproduces ``artifacts/ladder.json[label]``. The resample counts differ between the table
    intervals and the paired test because the recorded artifact does; the delta is unaffected,
    the interval and p-value are.
    """
    return Experiment(
        name=name or f"ladder.{label}",
        datasets=dataset_axis((label,)),
        features=ladder_feature_axis(),
        models=ladder_model_axis(),
        populations=ladder_population_axis(),
        splits=split_axis(),
        knobs=Knobs(
            seed=config.SEED,
            budgets=ladder.BUDGETS,
            n_bootstrap=1000,
            n_bootstrap_paired=ladder.N_BOOTSTRAP,
            candidates="full",
        ),
        comparisons=(
            Comparison(
                ROLE_MODEL,
                "semif_reranker",
                ladder.PROBE,
                note="baselines minus SemIf: a negative delta means SemIf is ahead",
            ),
        ),
        history=True,
        note="the traceability ladder: families removed one rung at a time",
    )


# --- reading a report back --------------------------------------------------


def semif_margins(
    report: RunReport,
    *,
    population: str = "starved141",
    reference: str = "semif_reranker",
    exclude: Sequence[str] = ("random",),
    ndigits: int = 4,
) -> dict:
    """SemIf's margin over the best classical selector, per rung and per budget.

    The ladder's headline quantity, computed from the report rather than recorded by the
    kernel: it is a *reading* of the sweep, and the layer's job was to make the sweep data. It
    rounds as the recorded artifact does, so a tie in the argmax breaks the same way.
    """
    tables: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for cell in report.cells:
        if cell.population != population:
            continue
        rung = cell.cell.name(ROLE_FEATURES)
        tables[rung][cell.cell.name(ROLE_MODEL)] = {
            f"{r['budget']:.2f}": round(r["recall"], ndigits) for r in cell.results
        }

    out: dict[str, dict] = {}
    for rung, table in tables.items():
        if reference not in table:
            out[rung] = {
                "unmeasured": f"{reference} has no measured cell in population {population!r}"
            }
            continue
        margins: dict[str, dict] = {}
        for key in sorted(table[reference]):
            classical = [n for n in table if n != reference and n not in exclude]
            best = max(classical, key=lambda n: table[n][key])
            margins[key] = {
                "best_classical": best,
                "best_recall": table[best][key],
                "semif_recall": table[reference][key],
                "semif_margin": round(table[reference][key] - table[best][key], ndigits),
            }
        out[rung] = margins
    return out


# --- CLI --------------------------------------------------------------------

ARMS: dict[str, object] = {
    "study": study_arm,
    "study.covered": lambda: study_arm(candidates="covered"),
    "ladder.mutmut": lambda: ladder_arm("mutmut"),
    "ladder.full": lambda: ladder_arm("full"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=sorted(ARMS))
    parser.add_argument("--out", default=None, help="output directory (default: artifacts/)")
    parser.add_argument(
        "--tiers",
        nargs="*",
        default=None,
        help="cost tiers to spend; the rest are reported unmeasured",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    experiment = ARMS[args.arm]()
    report = run(
        experiment,
        out_dir=args.out,
        tiers=args.tiers,
        save=not args.no_save,
        verbose=not args.quiet,
    )
    if args.arm.startswith("ladder"):
        margins = semif_margins(report)
        print("\nSemIf margin over the best classical selector (starved141):")
        for rung, table in margins.items():
            if "unmeasured" in table:
                print(f"  {rung}: {table['unmeasured']}")
                continue
            cells = "  ".join(f"b{k}={v['semif_margin']:+.3f}" for k, v in sorted(table.items()))
            print(f"  {rung:>14}: {cells}")


if __name__ == "__main__":
    main()
