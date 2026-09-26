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
    "SEED_SWEEP",
    "SEMIF_LADDER_CACHE",
    "SPARSE_BUDGETS",
    "SPARSE_THRESHOLDS",
    "STARVED_THRESHOLDS",
    "VARIATION_BUDGETS",
    "VARIATION_PROBE",
    "VARIATION_RESAMPLES",
    "VARIATION_TABLE_RESAMPLES",
    "dataset",
    "dataset_axis",
    "embed_arm",
    "embed_cache",
    "embed_model_axis",
    "instruction_arm",
    "instruction_cache",
    "instruction_model_axis",
    "instruction_names",
    "ladder_arm",
    "ladder_feature_axis",
    "ladder_model_axis",
    "ladder_population_axis",
    "ladder_populations",
    "ladder_selectors",
    "main",
    "model_axis",
    "population_axis",
    "redundancy_arm",
    "redundancy_model_axis",
    "rung_block",
    "run_study",
    "semif_margins",
    "sparse_arm",
    "split_axis",
    "starved_arm",
    "starved_cache",
    "starved_model_axis",
    "starved_seeds_arm",
    "structured_feature_axis",
    "study_arm",
    "variation_comparisons",
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
    names: Sequence["populations.Population | str"] = (),
    *,
    sparse: Sequence[int] = (),
) -> Axis:
    """Populations, by registry name or as values, plus parameterised sparse thresholds.

    A parameterised population is passed as a value, because it is not in the registry: the
    registry holds the study's named vocabulary, which is a different thing from the set of
    populations a particular sweep uses. A sparse threshold and a starvation threshold are both
    of the second kind -- the number is part of the population's identity, so it belongs to the
    element rather than to a global name table.
    """
    elements: list[Element] = []
    for spec in names:
        population = populations.resolve(spec)
        elements.append(
            constant(
                population.name,
                population,
                note="from populations.STUDY" if isinstance(spec, str) else population.note,
            )
        )
    for threshold in sparse:
        population = populations.low_pair_recurrence(threshold)
        elements.append(constant(population.name, population, note=population.note))
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


# --- the variation arms -----------------------------------------------------
#
# The ``variations`` driver's seven sections. Each is its own experiment rather than one grid,
# because they differ in the knobs a knob is allowed to differ in: ``candidates`` (the starved
# arms use the full pool, the instruction and redundancy arms the covered set) and ``budgets``.
# What they share is the dataset, the split and the feature block, which is what lets one score
# cache serve them all.
#
# Two sections are NOT here, and ``experiment.md`` §13 says why: ``p5_trained`` needs an
# evaluation window *inside* the held-out tail plus a NaN convention for unscored pairs, and
# ``p1_direct``'s cache is absent from the artifacts, so its numbers cannot be reproduced at all.

#: The variation arms' budget grid, probe budget and resample counts: four points, not the study
#: arm's six, and the table intervals and the paired tests use different counts.
VARIATION_BUDGETS: tuple[float, ...] = (0.01, 0.05, 0.1, 0.2)
VARIATION_PROBE = 0.05
VARIATION_TABLE_RESAMPLES = 1000
VARIATION_RESAMPLES = 2000

#: The starvation thresholds the starved arms are evaluated at. They are *nested* -- a killing
#: pair with at most 2 failures also has at most 5 -- which is why one cache scored for the wider
#: threshold covers the narrower population, and why the instruction sweep can report both for
#: the price of one.
STARVED_THRESHOLDS: tuple[int, ...] = (2, 5)

#: The model seeds the XGBoost refit is repeated under. Only the *model* seed varies: the run
#: seed fixes the imposed change order and the split, and the caches are keyed to that order.
SEED_SWEEP: tuple[int, ...] = (1, 2, 3, 4)


def starved_cache(max_failures: int) -> Path:
    """The SemIf cache for a starvation threshold.

    The ``<=5`` cache is assembled from the ``<=2`` one plus a scored delta rather than
    re-scoring pairs that already exist, which is sound because the populations nest.
    """
    if max_failures == 2:
        return config.ARTIFACTS / "semif_scores_starved2_full.jsonl"
    return config.ARTIFACTS / f"semif_scores_starved{max_failures}_full.jsonl"


def instruction_names() -> tuple[str, ...]:
    """The wording variants, in the recorded order, with the control first.

    The control is the study's own text-only cache: the point of the sweep is whether a
    *different* wording beats the question the other arms were measured against, so the control
    has to be that one rather than a re-scored copy of it.
    """
    from .semif_runner import INSTRUCTION_VARIANTS

    return ("default", *(n for n in sorted(INSTRUCTION_VARIANTS) if n != "default"))


def instruction_cache(name: str, max_failures: int) -> Path:
    if name == "default":
        return Path(config.SEMIF_SCORES_FILE)
    return config.ARTIFACTS / f"semif_scores_instr_{name}_starved{max_failures}.jsonl"


def embed_cache() -> Path:
    return config.ARTIFACTS / "embed_scores.npy"


def _cached(
    name: str,
    path: Path,
    *,
    tier: str = "cpu",
    note: str = "",
    loader=None,
) -> Element:
    """An element for a precomputed score matrix.

    Tier ``cpu``, not ``cache``: the *model* is as cheap as reading a file, but the cell still
    assembles features and sweeps metrics, so the cell's cost is not a file read.

    ``loader`` matters because a cache is not always a scored-pair log: the embedding baseline is
    a whole ``[n_changes, n_tests]`` matrix saved with ``numpy.save``, so reading it with the
    default loader is a decode error rather than a wrong number.
    """
    return Element(
        name,
        lambda binding, name=name, path=path, loader=loader: models.CachedScores(
            name, path, loader=loader
        ),
        tier=tier,
        note=note or f"precomputed scores from {path.name}",
    )


def _rank_average(name: str, parents: Sequence[models.Selector], candidates_mode: str) -> models.Selector:
    """A fitted-free rank average of parents that are already declared."""
    return models.RankAverageSelector(name, parents, candidates_mode=candidates_mode)


def starved_model_axis(max_failures: int, *, seeds: Sequence[int] = ()) -> Axis:
    """The starved arm's selector set, or the two-tree subset the seed sweep refits.

    The four XGBoost arms are a history x coverage decomposition with lexical always on, so the
    mechanism -- that the full candidate set is the only regime where ``covers_function``
    separates candidates from non-candidates -- is measured rather than asserted:

        struct_lex        history + coverage + BM25
        static_lex        coverage + BM25
        struct_nocov_lex  history + BM25
        static_nocov_lex  BM25 only

    ``seeds`` switches to the refit subset: the two strongest trees at each seed, plus the SemIf
    cache they are paired against. The seed is part of the element's identity because it is part
    of what the element computes.
    """
    if seeds:
        semif = models.CachedScores("semif_textonly_full", starved_cache(max_failures))
        elements: list[Element] = [
            _model_element(
                models.XGBoostSelector(
                    include_lexical=True, candidates_mode="full", seed=seed
                ),
                note=f"xgboost_struct_lex refit under model seed {seed}",
            )
            for seed in seeds
        ]
        elements.extend(
            _model_element(
                models.XGBoostSelector(
                    exclude_history=True,
                    include_lexical=True,
                    candidates_mode="full",
                    seed=seed,
                ),
                note=f"xgboost_static_lex refit under model seed {seed}",
            )
            for seed in seeds
        )
        elements.append(_model_element(semif))
        return Axis(ROLE_MODEL, tuple(elements), note="the seed sweep's refit subset")

    static_nocov_lex = models.XGBoostSelector(
        exclude_history=True, exclude_coverage=True, include_lexical=True, candidates_mode="full"
    )
    struct = models.XGBoostSelector(candidates_mode="full")
    semif = models.CachedScores("semif_textonly_full", starved_cache(max_failures))
    return Axis(
        ROLE_MODEL,
        (
            _model_element(models.RandomSelector()),
            _model_element(models.RecencySelector()),
            _model_element(models.FailureRateSelector()),
            _model_element(models.CoverageSelector()),
            _model_element(models.StructuralRuleSelector()),
            _model_element(models.LexicalSelector()),
            _model_element(static_nocov_lex),
            _model_element(
                models.XGBoostSelector(
                    exclude_history=True, include_lexical=True, candidates_mode="full"
                )
            ),
            _model_element(
                models.XGBoostSelector(
                    exclude_coverage=True, include_lexical=True, candidates_mode="full"
                )
            ),
            _model_element(struct),
            _model_element(models.XGBoostSelector(include_lexical=True, candidates_mode="full")),
            _model_element(semif),
            _model_element(
                _rank_average("rankaverage_xgb_semif", (static_nocov_lex, semif), "full")
            ),
        ),
        note="the starved arm's selectors; the two caches are the same model under one wording",
    )


def instruction_model_axis(max_failures: int) -> Axis:
    """BM25, the strongest cheap tree, and one SemIf element per wording variant."""
    elements = [
        _model_element(models.LexicalSelector()),
        _model_element(
            models.XGBoostSelector(
                exclude_history=True, exclude_coverage=True, include_lexical=True,
                candidates_mode="covered",
            )
        ),
    ]
    elements.extend(
        _cached(f"semif_{name}", instruction_cache(name, max_failures))
        for name in instruction_names()
    )
    return Axis(ROLE_MODEL, tuple(elements), note="the instruction-wording sweep")


def embed_model_axis(candidates: str) -> Axis:
    """The embedding baseline's parents: the code encoder against BM25 and two cheap rules."""
    elements = [
        _model_element(models.LexicalSelector()),
        _model_element(models.CoverageSelector()),
        _model_element(models.StructuralRuleSelector()),
        _cached("embed_codebert", embed_cache(), loader=models.load_matrix),
    ]
    if candidates == "covered":
        elements.insert(1, _cached("semif_textonly", Path(config.SEMIF_SCORES_FILE)))
        elements.append(
            _model_element(
                models.XGBoostSelector(
                    exclude_history=True, exclude_coverage=True, include_lexical=True,
                    candidates_mode="covered",
                )
            )
        )
    order = (
        ("bm25_lexical", "semif_textonly", "embed_codebert", "xgboost_static_nocov_lex")
        if candidates == "covered"
        else ("bm25_lexical", "embed_codebert", "structural_rule", "coverage")
    )
    by_name = {e.name: e for e in elements}
    return Axis(ROLE_MODEL, tuple(by_name[name] for name in order), note="the embedding baseline")


def redundancy_model_axis() -> Axis:
    """P5's parents plus the two rank averages that test whether SemIf is redundant."""
    static_nocov_lex = models.XGBoostSelector(
        exclude_history=True, exclude_coverage=True, include_lexical=True, candidates_mode="covered"
    )
    struct = models.XGBoostSelector(candidates_mode="covered")
    semif = models.CachedScores("semif_textonly", Path(config.SEMIF_SCORES_FILE))
    return Axis(
        ROLE_MODEL,
        (
            _model_element(models.RandomSelector()),
            _model_element(models.RecencySelector()),
            _model_element(models.FailureRateSelector()),
            _model_element(models.CoverageSelector()),
            _model_element(models.StructuralRuleSelector()),
            _model_element(models.LexicalSelector()),
            _model_element(static_nocov_lex),
            _model_element(struct),
            _model_element(semif),
            _model_element(
                _rank_average("rankaverage_xgb_semif", (static_nocov_lex, semif), "covered")
            ),
            _model_element(
                _rank_average("rankaverage_xgb_struct_semif", (struct, semif), "covered")
            ),
        ),
        note="the redundancy test: two trees, the text model, and both rank averages",
    )


def _variation_knobs(candidates: str) -> Knobs:
    return Knobs(
        seed=config.SEED,
        budgets=VARIATION_BUDGETS,
        n_bootstrap=VARIATION_TABLE_RESAMPLES,
        n_bootstrap_paired=VARIATION_RESAMPLES,
        candidates=candidates,
    )


def variation_comparisons(references: Sequence[str]) -> tuple[Comparison, ...]:
    """Every reference paired at every budget.

    A comparison is defined by one probe budget, and the recorded artifacts pair at all four of
    the variation grid -- because a lever that helps only at a low budget is a different finding
    from one that helps throughout. So the pairs are declared rather than the renderer computing
    a second family of statistics outside the report.
    """
    return tuple(
        Comparison(ROLE_MODEL, reference, budget)
        for reference in references
        for budget in VARIATION_BUDGETS
    )


def starved_arm(max_failures: int = 2, *, name: str | None = None) -> Experiment:
    """The full-candidate starved arm: the positive result, re-measured without the crutch."""
    return Experiment(
        name=name or f"starved{max_failures}",
        datasets=dataset_axis(("mutmut",)),
        features=structured_feature_axis(),
        models=starved_model_axis(max_failures),
        populations=population_axis((populations.starved(max_failures),)),
        splits=split_axis(),
        knobs=_variation_knobs("full"),
        comparisons=variation_comparisons(
            (
                "xgboost_static_nocov_lex",
                "structural_rule",
                "xgboost_struct_lex",
                "xgboost_static_lex",
            )
        ),
        history=True,
        note="the starved arm over the full candidate pool",
    )


def starved_seeds_arm(
    max_failures: int = 2, *, model_seed: int, name: str | None = None
) -> Experiment:
    """One refit of the starved arm's two strongest trees, under its own model seed.

    A *model* seed, not a run seed: varying ``seed`` would move the imposed change order and the
    split, and the score caches are keyed to that order, so it would invalidate the paired
    comparison instead of testing it.
    """
    threshold = populations.starved(max_failures)
    knobs = _variation_knobs("full")
    return Experiment(
        name=name or f"starved{max_failures}.seed{model_seed}",
        datasets=dataset_axis(("mutmut",)),
        features=structured_feature_axis(),
        models=starved_model_axis(max_failures, seeds=(model_seed,)),
        populations=population_axis((threshold,)),
        splits=split_axis(),
        knobs=replace(knobs, model_seed=model_seed),
        # The trees are the references, so the SemIf cell is the one paired -- and the layer
        # computes ``cell - reference``, which is the recorded convention ("SemIf minus the
        # tree") without the renderer having to flip a sign.
        comparisons=variation_comparisons(("xgboost_struct_lex", "xgboost_static_lex")),
        history=True,
        note=f"the starved arm refit under model seed {model_seed}",
    )


def instruction_arm(
    max_failures: int = 5,
    *,
    secondary: int = 2,
    name: str | None = None,
) -> Experiment:
    """The instruction-wording sweep, over both starvation thresholds at once.

    Both thresholds are population elements in one run, which is free: the caches are scored for
    the wider threshold and the narrower population is a subset of it.
    """
    thresholds = [populations.starved(max_failures)]
    if secondary != max_failures:
        thresholds.append(populations.starved(secondary))
    return Experiment(
        name=name or f"instruction{max_failures}",
        datasets=dataset_axis(("mutmut",)),
        features=structured_feature_axis(),
        models=instruction_model_axis(max_failures),
        populations=population_axis(tuple(thresholds)),
        splits=split_axis(),
        knobs=_variation_knobs("covered"),
        comparisons=variation_comparisons(("semif_default",)),
        history=True,
        note="the instruction sweep: five wordings against the one the study used",
    )


def embed_arm(candidates: str = "covered", *, name: str | None = None) -> Experiment:
    """The code-embedding baseline, at one candidate mode.

    Two arms rather than one because the full-pool comparison is the one where a semantic ranker
    could pay off -- the covered mask is the crutch that lets structure win cheaply -- and
    ``candidates`` is a knob.
    """
    chosen: list[populations.Population] = []
    if candidates == "covered":
        chosen.append(populations.FAULT_BEARING)
    chosen.append(populations.starved(5))
    return Experiment(
        name=name or f"embed.{candidates}",
        datasets=dataset_axis(("mutmut",)),
        features=structured_feature_axis(),
        models=embed_model_axis(candidates),
        populations=Axis(
            ROLE_POPULATION,
            tuple(constant(p.name, p) for p in chosen),
            note="all held-out faults, and the starved subset",
        ),
        splits=split_axis(),
        knobs=_variation_knobs(candidates),
        comparisons=variation_comparisons(("bm25_lexical",)),
        history=True,
        note="the code-embedding baseline against BM25 and two cheap structural rules",
    )


def redundancy_arm(*, name: str | None = None) -> Experiment:
    """P5: does SemIf carry signal the cheap structured model does not already have?

    The test is not a correlation but a *parent-versus-child* comparison: if the second model
    adds independent signal the rank average beats both parents, and if it is redundant the
    average sits between them. Three references are declared because the pairs the write-up
    needs span two of them -- the averages are compared against their own structural parent as
    well as against each other's -- and a comparison is defined by its reference.
    """
    return Experiment(
        name=name or "redundancy",
        datasets=dataset_axis(("mutmut",)),
        features=structured_feature_axis(),
        models=redundancy_model_axis(),
        populations=population_axis(("fault_bearing",)),
        splits=split_axis(),
        knobs=_variation_knobs("covered"),
        comparisons=variation_comparisons(
            ("xgboost_static_nocov_lex", "semif_textonly", "xgboost_struct")
        ),
        history=True,
        note="is the transformer redundant given cheap structure?",
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
