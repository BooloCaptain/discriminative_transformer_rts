"""The traceability ladder's declarations, and the condition that runs them.

The ladder removes feature families one rung at a time and asks whether SemIf crosses the
classical baselines as traceability is withdrawn. Its declarations live here rather than in
:mod:`rts.render.ladder` because the driver is a renderer: it runs this condition and writes the recorded
artifact's shape.

``rts.render.ladder`` must not be imported from here. The driver imports this package, so a submodule
importing it back would close a cycle.
"""

from __future__ import annotations

import json

import numpy as np

from .. import config, features
from ..data import subsets
from ..data.contract import Dataset, Undefined
from ..experiment import (
    FACTOR_FEATURES,
    FACTOR_MODEL,
    FACTOR_SUBSET,
    Contrast,
    Controls,
    Experiment,
    Factor,
    Level,
    constant,
)
from ..model import rankers
from .factors import dataset_factor, model_factor, split_factor

# --- the traceability ladder's declarations --------------------------------

#: The ladder's own budget grid and probe budget: four points, not the study condition's six.
LADDER_BUDGETS: tuple[float, ...] = (0.01, 0.05, 0.1, 0.2)
LADDER_PROBE = 0.05
#: Resamples for the rung tables and for the paired tests. Different, because a table's interval
#: and a paired p-value are two quantities with two precision needs, and the recorded artifact
#: used different counts for them.
LADDER_TABLE_RESAMPLES = 1000
LADDER_RESAMPLES = 2000

#: SemIf scores over the full candidate pool, cached for the ladder's held-out subset.
#: The ``141`` in the name is provenance only: under corrected full-suite labels the cold start
#: filter collapses, so the subset is simply "the changes this cache covers".
SEMIF_LADDER_CACHE = config.ARTIFACTS / "semif_scores_ladder141_full.jsonl"

#: The rungs, cumulatively. A rung name states what is *unavailable* at that rung, which is what
#: makes the degradation curve readable.
RUNGS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("L0_all", ()),
    ("L1_notemporal", ("temporal",)),
    ("L2_nocoverage", ("temporal", "coverage")),
    ("L3_noproximity", ("temporal", "coverage", "proximity")),
)


def rung_block(removed: tuple[str, ...]) -> features.FeatureBlock:
    """The structured block with the rung's families withheld.

    Withholding goes through the block's own ``without_families``, so a rung takes the same
    undefined path a genuinely absent capability takes and a typo in a family name raises
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


def ladder_subsets(ds: Dataset) -> dict[str, subsets.Subset | Undefined]:
    """The ladder's two averaging subsets, built from the dataset.

    ``cache_covered`` is defined by which changes the SemIf cache covers, so when the cache is
    absent it is **unavailable** rather than smaller -- reporting fewer pairs as if they were the
    subset would be a different claim. ``held_out`` is defined by the labels alone.
    """
    out: dict[str, subsets.Subset | Undefined] = {}

    cached = _cached_change_ids()
    if cached is None:
        out["cache_covered"] = Undefined(
            requirement="artifact:semif_ladder_cache",
            note=(
                f"{SEMIF_LADDER_CACHE.name} is missing, so the cache_covered subset cannot "
                "exist; the paired contrast has no SemIf scores to pair against"
            ),
        )
    else:
        ids = [ds.change_id(c) for c in ds.changes]
        mask = np.array([cid in cached for cid in ids], dtype=bool)
        out["cache_covered"] = subsets.Subset(
            name="cache_covered",
            note=(
                "held-out changes with a complete full-pool SemIf cache; paired contrasts "
                "happen here"
            ),
            needs=("labels",),
            predicate=lambda _material, mask=mask: mask,
        )

    out["held_out"] = subsets.Subset(
        name="held_out",
        note="every held-out fault-bearing change",
        needs=("labels",),
        predicate=lambda inputs: inputs["faults"],
    )
    return out


def ladder_feature_factor() -> Factor:
    """The ladder's rungs, as blocks with families withheld.

    The rung *definition* is a study choice and stays in ``rts.render.ladder``; what the layer does is
    turn each one into a level. Withholding a family goes through the block's own
    ``without_families``, so a rung takes the same undefined path a genuinely absent
    capability takes and a typo raises instead of quietly ablating nothing.
    """
    levels = []
    for name, removed in RUNGS:
        block = rung_block(removed)
        levels.append(
            constant(
                name,
                block,
                tier="cpu",
                note=f"withheld families: {', '.join(removed) if removed else 'none'}",
            )
        )
    return Factor(FACTOR_FEATURES, tuple(levels), note="the traceability ladder's rungs")


def ladder_rankers() -> list[rankers.Ranker]:
    """The ladder's ranker set.

    A study choice, so it lives with the config rather than being read back out of the driver.
    The two lists must stay in step until ``ladder.py`` is deleted (``docs/experiment.md`` §12).
    """
    return [
        rankers.RandomRanker(),
        rankers.RecencyRanker(),
        rankers.FailureRateRanker(),
        rankers.CoverageRanker(),
        rankers.StructuralRuleRanker(),
        rankers.LexicalRanker(),
        rankers.XGBoostRanker(include_lexical=True),
        rankers.XGBoostRanker(include_lexical=False),
    ]


def _ladder_semif_applies(binding) -> Undefined | None:
    """Why the ladder's SemIf cache does not apply to this design point's subset.

    The cache was built over the changes ``cache_covered`` selects, and ``semif.load_scores``
    fills every *uncached* pair with a sentinel rather than reporting that it has no score. So
    evaluating SemIf on the other subset yields a number that looks like a measurement and
    is not. ``ladder.py`` avoided that by simply not tabulating SemIf there; declaring the
    inapplicability keeps the design point in the report, as a finding with a reason.
    """
    subset = binding.factors.get(FACTOR_SUBSET)
    if subset == "cache_covered":
        return None
    return Undefined(
        requirement="artifact:semif_ladder_coverage",
        note=(
            "the ladder's SemIf cache covers only the cache_covered changes, so it has no scores "
            f"for the {subset!r} subset"
        ),
    )


def ladder_model_factor(include_semif: bool = True) -> Factor:
    factor = model_factor(ladder_rankers())
    if not include_semif:
        return factor
    return factor.extend(
        Level(
            "semif_reranker",
            lambda _binding: rankers.SemIfRanker(scores_file=SEMIF_LADDER_CACHE),
            note="SemIf scores over the full pool, cached for the ladder's held-out subset",
            applies=_ladder_semif_applies,
        )
    )


def ladder_subset_factor() -> Factor:
    """The ladder's two subsets, built from the dataset by the ladder's own definition.

    ``build_populations`` returns an :class:`Undefined` when the SemIf cache is absent, which
    the layer passes through unchanged: the subset is then unavailable rather than smaller,
    which is the distinction the ladder was written to preserve.
    """

    def level(name: str) -> Level:
        return Level(
            name,
            lambda binding, name=name: ladder_subsets(binding.dataset)[name],
            tier="cpu",
            note=f"the ladder's {name} subset, or why it cannot exist",
        )

    return Factor(
        FACTOR_SUBSET,
        (level("held_out"), level("cache_covered")),
        note="held-out change sets: one for every fault, one for the cached subset",
    )


def ladder_condition(label: str = "mutmut", *, name: str | None = None) -> Experiment:
    """The traceability ladder: four rungs, two subsets, paired against SemIf.

    Reproduces ``artifacts/ladder.json[label]``. The resample counts differ between the table
    intervals and the paired test because the recorded artifact does; the delta is unaffected,
    the interval and p-value are.
    """
    return Experiment(
        name=name or f"ladder.{label}",
        datasets=dataset_factor((label,)),
        features=ladder_feature_factor(),
        models=ladder_model_factor(),
        subsets=ladder_subset_factor(),
        splits=split_factor(),
        controls=Controls(
            seed=config.SEED,
            budgets=LADDER_BUDGETS,
            n_bootstrap=LADDER_TABLE_RESAMPLES,
            n_bootstrap_paired=LADDER_RESAMPLES,
            candidate_policy="full",
        ),
        contrasts=(
            Contrast(
                FACTOR_MODEL,
                "semif_reranker",
                LADDER_PROBE,
                note="baselines minus SemIf: a negative delta means SemIf is ahead",
            ),
        ),
        temporal=True,
        note="the traceability ladder: families removed one rung at a time",
    )
