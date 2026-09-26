"""The study's experiments, expressed as values: axes by role, and the named arms.

This is the config module the experiment layer was built for. It holds the study's *choices*
-- which datasets, which rungs, which selectors, which populations, which budgets -- and
nothing about how a sweep is executed.

**Axis builders are functions, not module constants**, wherever the elements hold state. An
:class:`~rts.experiment.Experiment` is immutable and could be a constant, but its axes hold
selector instances that train on use, so a fresh axis per arm keeps two runs from sharing one
model object.

**Two arms here are verification vehicles.** ``study_arm()``/``sparse_arm()`` reproduce
``artifacts/results_{full,covered}.json`` and ``ladder_arm()`` reproduces
``artifacts/ladder.json``, both through ``scripts/verify_experiment_layer.py``. The drivers that
render those artifacts -- ``rts/pipeline.py`` and ``rts/ladder.py`` -- are now thin: they run an
arm declared here and write it in the shape the recorded numbers are written against. The
remaining drivers (``variations``, ``bundles``, ``bugsinpy``, ``analysis``) still hold their own
sweeps and are the next migration (``experiment.md`` §13).

Usage::

    python -m rts.studies study
    python -m rts.studies ladder.mutmut --tiers cpu
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np

from . import bundles, config, datasets, features, models, populations, splits
from .contract import Dataset, Unmeasured
from .experiment import (
    ROLE_DATASET,
    ROLE_FEATURES,
    ROLE_MODEL,
    ROLE_POPULATION,
    ROLE_SPLIT,
    Axis,
    Binding,
    Comparison,
    Element,
    Environment,
    Experiment,
    Knobs,
    RunReport,
    constant,
    run,
)

__all__ = [
    "ABLATION_MODELS",
    "ARMS",
    "LADDER_BUDGETS",
    "LADDER_PROBE",
    "LADDER_RESAMPLES",
    "LADDER_TABLE_RESAMPLES",
    "RUNGS",
    "SEMIF_LADDER_CACHE",
    "SPARSE_BUDGETS",
    "SPARSE_THRESHOLDS",
    "dataset",
    "dataset_axis",
    "ladder_arm",
    "ladder_feature_axis",
    "ladder_model_axis",
    "ladder_population_axis",
    "ladder_populations",
    "ladder_selectors",
    "main",
    "model_axis",    "population_axis",
    "rung_block",
    "run_study",
    "semif_margins",
    "sparse_arm",
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


# --- the traceability ladder's declarations --------------------------------

#: The ladder's own budget grid and probe budget: four points, not the study arm's six.
LADDER_BUDGETS: tuple[float, ...] = (0.01, 0.05, 0.1, 0.2)
LADDER_PROBE = 0.05
#: Resamples for the rung tables and for the paired tests. Different, because a table's interval
#: and a paired p-value are two quantities with two precision needs, and the recorded artifact
#: used different counts for them.
LADDER_TABLE_RESAMPLES = 1000
LADDER_RESAMPLES = 2000

#: SemIf scores over the full candidate pool, cached for the ladder's held-out subset.
#: The ``141`` in the name is provenance only: under corrected full-suite labels the starved
#: filter collapses, so the population is simply "the changes this cache covers".
SEMIF_LADDER_CACHE = config.ARTIFACTS / "semif_scores_ladder141_full.jsonl"

#: The rungs, cumulatively. A rung name states what is *unavailable* at that rung, which is what
#: makes the degradation curve readable.
RUNGS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("L0_all", ()),
    ("L1_nohistory", ("history",)),
    ("L2_nocoverage", ("history", "coverage")),
    ("L3_notrace", ("history", "coverage", "traceability")),
)


def rung_block(removed: tuple[str, ...]) -> features.FeatureBlock:
    """The structured block with the rung's families withheld.

    Withholding goes through the block's own ``without_families``, so a rung takes the same
    unmeasured path a genuinely absent capability takes and a typo in a family name raises
    instead of quietly ablating nothing -- which is the bug that motivated the refactor.
    """
    return (
        features.STRUCTURED
        if not removed
        else features.STRUCTURED.without_families(*removed)
    )


def _cached_change_ids() -> set[str] | None:
    """Change ids with a complete full-pool SemIf cache, or ``None`` if there is none."""
    if not SEMIF_LADDER_CACHE.exists():
        return None
    ids: set[str] = set()
    with SEMIF_LADDER_CACHE.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["change_id"])
    return ids


def ladder_populations(ds: Dataset) -> dict[str, "populations.Population | Unmeasured"]:
    """The ladder's two averaging populations, built from the dataset.

    ``starved141`` is defined by which changes the SemIf cache covers, so when the cache is
    absent it is **unavailable** rather than smaller -- reporting fewer pairs as if they were the
    population would be a different claim. ``heldout530`` is defined by the labels alone.
    """
    out: dict[str, populations.Population | Unmeasured] = {}

    cached = _cached_change_ids()
    if cached is None:
        out["starved141"] = Unmeasured(
            requirement="artifact:semif_ladder_cache",
            note=(
                f"{SEMIF_LADDER_CACHE.name} is missing, so the starved141 population cannot "
                "exist; the paired comparison has no SemIf scores to pair against"
            ),
        )
    else:
        ids = [ds.change_id(c) for c in ds.changes]
        mask = np.array([cid in cached for cid in ids], dtype=bool)
        out["starved141"] = populations.Population(
            name="starved141",
            note=(
                "held-out changes with a complete full-pool SemIf cache; paired comparisons "
                "happen here"
            ),
            needs=("labels",),
            predicate=lambda _material, mask=mask: mask,
        )

    out["heldout530"] = populations.Population(
        name="heldout530",
        note="every held-out fault-bearing change",
        needs=("labels",),
        predicate=lambda material: material["faults"],
    )
    return out


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
    for name, removed in RUNGS:
        block = rung_block(removed)
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


#: The BM25 ablation probes from ``plan.md``: shuffle the change text, the test text, or both,
#: and re-score the same pairs. They are *controls* rather than competitors, but they are
#: produced by the same machinery from the same context, so they are model elements instead of a
#: driver's second loop with its own bookkeeping.
ABLATION_MODELS: tuple[tuple[str, dict], ...] = (
    ("bm25_change_shuffled", {"shuffle_changes": True}),
    ("bm25_test_shuffled", {"shuffle_tests": True}),
    ("bm25_both_shuffled", {"shuffle_changes": True, "shuffle_tests": True}),
)

#: The sparse-arm thresholds. ``max_pair_count`` is a *population* parameter, so a threshold is a
#: population element rather than a knob. The *budgets* are a knob, though, and the sparse arm
#: reports three where the headline arm reports six -- so the sparse arm is a second experiment
#: rather than a corner of the first grid. That is a real consequence of the design and it is
#: recorded here rather than hidden: a metric grid cannot be a factor while the sweep is
#: vectorised across budgets.
SPARSE_THRESHOLDS: tuple[int, ...] = (80, 160)
SPARSE_BUDGETS: tuple[float, ...] = (0.05, 0.1, 0.2)


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


def model_axis(
    selectors: Sequence[models.Selector] | None = None,
    *,
    ablations: bool = False,
    include_semif: bool = True,
) -> Axis:
    chosen = (
        list(selectors)
        if selectors is not None
        else models.default_selectors(include_semif=include_semif)
    )
    if ablations:
        chosen.extend(
            models.LexicalSelector(**kwargs) for _, kwargs in ABLATION_MODELS
        )
    return Axis(
        ROLE_MODEL,
        tuple(_model_element(s) for s in chosen),
        note="the study's selector set" + (", plus the BM25 shuffle controls" if ablations else ""),
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
            lambda _binding: models.SemIfSelector(scores_file=SEMIF_LADDER_CACHE),
            note="SemIf scores over the full pool, cached for the ladder's held-out subset",
            applies=_ladder_semif_applies,
        )
    )


# --- populations ------------------------------------------------------------


def population_axis(
    names: Sequence[str] = (),
    *,
    sparse: Sequence[int] = (),
) -> Axis:
    """Named populations from the study's registry, plus parameterised sparse thresholds.

    A sparse threshold is a population, not a knob: it changes which changes a metric is
    averaged over, which is exactly what a population is, and putting it here makes the sparse
    arm an ordinary corner of the grid.
    """
    elements = [
        constant(name, populations.population(name), note="from populations.STUDY")
        for name in names
    ]
    for threshold in sparse:
        population = populations.low_pair_recurrence(threshold)
        elements.append(
            constant(
                population.name,
                population,
                note=f"changes whose (file, test) pairs all recur at most {threshold} times",
            )
        )
    return Axis(
        ROLE_POPULATION,
        tuple(elements),
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
            lambda binding, name=name: ladder_populations(binding.dataset)[name],
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
    ablations: bool = True,
    include_semif: bool = True,
    name: str | None = None,
) -> Experiment:
    """The study's headline arm: every selector, one dataset, the BM25 shuffle controls.

    With ``candidates="full"`` and ``label="mutmut"`` this reproduces
    ``artifacts/results_full.json``. The shuffle controls are part of the same grid because they
    share the dataset, the split, the feature block, the candidate pool and the budgets -- only
    the model differs, and that is the role they are.
    """
    return Experiment(
        name=name or f"study.{label}.{candidates}",
        datasets=dataset_axis((label,)),
        features=structured_feature_axis(),
        models=model_axis(ablations=ablations, include_semif=include_semif),
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


def sparse_arm(
    label: str = "mutmut",
    *,
    thresholds: Sequence[int] = SPARSE_THRESHOLDS,
    budgets: Sequence[float] = SPARSE_BUDGETS,
    candidates: str = "full",
    include_semif: bool = True,
    name: str | None = None,
) -> Experiment:
    """The sparse arm: the same selectors over the rarely-recurring corners of the data.

    Its own experiment rather than a population of :func:`study_arm`, because it reports three
    budgets where the headline arm reports six and ``budgets`` is a knob. The selectors are the
    headline set without the shuffle controls, matching what the recorded artifact tabulates.
    """
    return Experiment(
        name=name or f"sparse.{label}.{candidates}",
        datasets=dataset_axis((label,)),
        features=structured_feature_axis(),
        models=model_axis(models.default_selectors(include_semif=include_semif)),
        populations=population_axis((), sparse=thresholds),
        splits=split_axis(),
        knobs=Knobs(
            seed=config.SEED,
            budgets=tuple(budgets),
            n_bootstrap=config.DEFAULT_BOOTSTRAP,
            candidates=candidates,
        ),
        history=True,
        note="the sparse arm: (file, test) pairs that rarely recur",
    )


def run_study(
    label: str = "mutmut",
    *,
    candidates: str = "full",
    ablations: bool = True,
    include_semif: bool = True,
    sparse: Sequence[int] = SPARSE_THRESHOLDS,
    out_dir: Path | str | None = None,
    knobs_overrides: dict | None = None,
    save: bool = True,
    verbose: bool = True,
) -> tuple[RunReport, RunReport | None]:
    """Run the headline arm and, unless ``sparse`` is empty, the sparse arm.

    Two experiments, one score cache: the second run's contexts differ from the first's only in
    which rows the metric averages, so it reuses every score matrix and only re-sweeps the
    metrics. Returns ``(headline, sparse_or_None)``.

    ``knobs_overrides`` applies to both arms' declared knobs *individually*, so an override can
    change the seed or the resample count without disturbing the budgets that make the two arms
    different.
    """
    scores: dict = {}

    def effective(experiment: Experiment) -> Knobs:
        if not knobs_overrides:
            return experiment.knobs
        return replace(experiment.knobs, **knobs_overrides)

    headline_experiment = study_arm(
        label, candidates=candidates, ablations=ablations, include_semif=include_semif
    )
    headline = run(
        headline_experiment,
        out_dir=out_dir,
        knobs=effective(headline_experiment),
        scores=scores,
        save=save,
        verbose=verbose,
    )
    if not sparse:
        return headline, None
    sparse_experiment = sparse_arm(
        label, thresholds=sparse, candidates=candidates, include_semif=include_semif
    )
    secondary = run(
        sparse_experiment,
        out_dir=out_dir,
        knobs=effective(sparse_experiment),
        scores=scores,
        save=False,
        verbose=verbose,
    )
    return headline, secondary


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
            budgets=LADDER_BUDGETS,
            n_bootstrap=LADDER_TABLE_RESAMPLES,
            n_bootstrap_paired=LADDER_RESAMPLES,
            candidates="full",
        ),
        comparisons=(
            Comparison(
                ROLE_MODEL,
                "semif_reranker",
                LADDER_PROBE,
                note="baselines minus SemIf: a negative delta means SemIf is ahead",
            ),
        ),
        history=True,
        note="the traceability ladder: families removed one rung at a time",
    )


def dataset(label: str = "mutmut", *, knobs: Knobs | None = None, **shared) -> Dataset:
    """The built dataset for a label source, for a caller that needs the data itself.

    For an adapter rendering a dataset description, or a check computing a statistic, rather
    than for a sweep over it. Rebuilding is cheap and explicitly permitted: two datasets may
    coexist in one process (``refactor.md`` §7), so a consumer does not have to reach into a
    run's internals to get at what it measured.
    """
    element = dataset_axis((label,)).get(f"marshmallow_{label}")
    value = element.build(Binding(env=Environment(knobs=knobs or Knobs(), shared=dict(shared))))
    if not isinstance(value, Dataset):
        raise TypeError(f"dataset element {element.name!r} did not build a Dataset: {value!r}")
    return value


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
