"""The traceability ladder's declarations, and the arm that runs them.

The ladder removes feature families one rung at a time and asks whether SemIf crosses the
classical baselines as traceability is withdrawn. Its declarations live here rather than in
:mod:`rts.ladder` because the driver is a renderer: it runs this arm and writes the recorded
artifact's shape.

``rts.ladder`` must not be imported from here. The driver imports this package, so a submodule
importing it back would close a cycle.
"""

from __future__ import annotations

import json

import numpy as np

from .. import config, features
from ..data import populations
from ..data.contract import Dataset, Unmeasured
from ..experiment import (
    ROLE_FEATURES,
    ROLE_MODEL,
    ROLE_POPULATION,
    Axis,
    Comparison,
    Element,
    Experiment,
    Knobs,
    constant,
)
from ..model import selectors
from .axes import dataset_axis, model_axis, split_axis

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


def ladder_selectors() -> list[selectors.Selector]:
    """The ladder's selector set.

    A study choice, so it lives with the config rather than being read back out of the driver.
    The two lists must stay in step until ``ladder.py`` is deleted (``docs/experiment.md`` §12).
    """
    return [
        selectors.RandomSelector(),
        selectors.RecencySelector(),
        selectors.FailureRateSelector(),
        selectors.CoverageSelector(),
        selectors.StructuralRuleSelector(),
        selectors.LexicalSelector(),
        selectors.XGBoostSelector(include_lexical=True),
        selectors.XGBoostSelector(include_lexical=False),
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
            lambda _binding: selectors.SemIfSelector(scores_file=SEMIF_LADDER_CACHE),
            note="SemIf scores over the full pool, cached for the ladder's held-out subset",
            applies=_ladder_semif_applies,
        )
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
