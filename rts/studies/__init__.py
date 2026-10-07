"""The study's experiments, expressed as values: factors by role, and the named conditions.

This is the config package the experiment layer was built for. It holds the study's *choices*
-- which datasets, which rungs, which rankers, which subsets, which budgets -- and
nothing about how a sweep is executed.

It used to be one module of about 1150 lines holding three catalogues, so the variation conditions
were not findable without reading all of it. It is now a package, and the public names are
re-exported here, so ``studies.study_condition()``, ``studies.RUNGS`` and ``studies.semif_margins``
read exactly as they did:

* :mod:`.factors` -- the role-by-role level builders (datasets, feature blocks, ranker sets,
  subsets, split);
* :mod:`.conditions` -- the named conditions and the runner that shares one score cache between them;
* :mod:`.ladder` -- the traceability ladder's declarations and its condition;
* :mod:`.variations` -- the variation study's conditions;
* :mod:`.readings` -- quantities computed *from* a report rather than recorded by the kernel.

**Factor builders are functions, not module constants**, wherever the levels hold state. An
:class:`~rts.experiment.Experiment` is immutable and could be a constant, but its factors hold
ranker instances that train on use, so a fresh factor per condition keeps two runs from sharing one
model object.

**Two conditions here are verification vehicles.** ``study_condition()``/``low_cooccurrence_condition()`` reproduce
``artifacts/results_{full,covered}.json`` and ``ladder_condition()`` reproduces
``artifacts/ladder.json``, both through ``scripts/verify_experiment_layer.py``. The drivers that
render those artifacts -- ``rts/render/pipeline.py``, ``rts/render/ladder.py`` and ``rts/render/bugsinpy.py`` -- are
thin: they run a condition declared here and write it in the shape the recorded numbers are written
against. ``rts/bundles.py`` is the last driver that still holds its own sweep and is therefore
the next migration (``docs/experiment.md`` §13).

Usage::

    python -m rts.studies study
    python -m rts.studies ladder.mutmut --tiers cpu
"""

from __future__ import annotations

import argparse

from ..experiment import run
from . import bugsinpy
from .bugsinpy import BUGSINPY_BUDGETS, bugsinpy_condition, run_bugsinpy
from .conditions import (
    LOW_COOCCURRENCE_BUDGETS,
    LOW_COOCCURRENCE_THRESHOLDS,
    dataset,
    low_cooccurrence_condition,
    run_study,
    semif_production_condition,
    study_condition,
)
from .factors import (
    ABLATION_MODELS,
    dataset_factor,
    model_factor,
    semif_scoring_model_factor,
    split_factor,
    structured_feature_factor,
    subset_factor,
)
from .ladder import (
    LADDER_BUDGETS,
    LADDER_PROBE,
    LADDER_RESAMPLES,
    LADDER_TABLE_RESAMPLES,
    RUNGS,
    SEMIF_LADDER_CACHE,
    ladder_condition,
    ladder_feature_factor,
    ladder_model_factor,
    ladder_rankers,
    ladder_subset_factor,
    ladder_subsets,
    rung_block,
)
from .readings import semif_margins
from .variations import (
    COLD_START_THRESHOLDS,
    SEED_SWEEP,
    VARIATION_BUDGETS,
    VARIATION_PROBE,
    VARIATION_RESAMPLES,
    VARIATION_TABLE_RESAMPLES,
    cold_start_cache,
    cold_start_condition,
    cold_start_model_factor,
    cold_start_seeds_condition,
    embed_cache,
    embed_model_factor,
    embedding_condition,
    instruction_cache,
    instruction_condition,
    instruction_model_factor,
    instruction_names,
    redundancy_condition,
    redundancy_model_factor,
    variation_contrasts,
)

__all__ = [
    "ABLATION_MODELS",
    "CONDITIONS",
    "BUGSINPY_BUDGETS",
    "LADDER_BUDGETS",
    "LADDER_PROBE",
    "LADDER_RESAMPLES",
    "LADDER_TABLE_RESAMPLES",
    "RUNGS",
    "SEED_SWEEP",
    "SEMIF_LADDER_CACHE",
    "LOW_COOCCURRENCE_BUDGETS",
    "LOW_COOCCURRENCE_THRESHOLDS",
    "COLD_START_THRESHOLDS",
    "VARIATION_BUDGETS",
    "VARIATION_PROBE",
    "VARIATION_RESAMPLES",
    "VARIATION_TABLE_RESAMPLES",
    "dataset",
    "dataset_factor",
    "bugsinpy",
    "bugsinpy_condition",
    "embedding_condition",
    "embed_cache",
    "embed_model_factor",
    "instruction_condition",
    "instruction_cache",
    "instruction_model_factor",
    "instruction_names",
    "ladder_condition",
    "ladder_feature_factor",
    "ladder_model_factor",
    "ladder_subset_factor",
    "ladder_subsets",
    "ladder_rankers",
    "main",
    "model_factor",
    "subset_factor",
    "redundancy_condition",
    "redundancy_model_factor",
    "rung_block",
    "run_bugsinpy",
    "run_study",
    "semif_margins",
    "semif_production_condition",
    "semif_scoring_model_factor",
    "low_cooccurrence_condition",
    "split_factor",
    "cold_start_condition",
    "cold_start_cache",
    "cold_start_model_factor",
    "cold_start_seeds_condition",
    "structured_feature_factor",
    "study_condition",
    "variation_contrasts",
]


# --- CLI --------------------------------------------------------------------

CONDITIONS: dict[str, object] = {
    "study": study_condition,
    "study.covered": lambda: study_condition(candidate_sets="coverage_restricted"),
    "ladder.mutmut": lambda: ladder_condition("mutmut"),
    "ladder.full": lambda: ladder_condition("full"),
    # Declared so score *production* is addressable like any other condition: the GPU tier is what
    # makes it score rather than report a undefined design point.
    "semif.produce": lambda: semif_production_condition(),
    # One budget, which is the condition's unit: the recorded intervals are per-budget, so the full
    # sweep is four runs through ``studies.run_bugsinpy`` (see ``rts/studies/bugsinpy``).
    "bugsinpy": lambda: bugsinpy_condition(0.05),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("condition", choices=sorted(CONDITIONS))
    parser.add_argument("--out", default=None, help="output directory (default: artifacts/)")
    parser.add_argument(
        "--tiers",
        nargs="*",
        default=None,
        help="cost tiers to spend; the rest are reported undefined",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    experiment = CONDITIONS[args.condition]()
    report = run(
        experiment,
        out_dir=args.out,
        tiers=args.tiers,
        save=not args.no_save,
        verbose=not args.quiet,
    )
    if args.condition.startswith("ladder"):
        margins = semif_margins(report)
        print("\nSemIf margin over the best classical ranker (cache_covered):")
        for rung, table in margins.items():
            if "undefined" in table:
                print(f"  {rung}: {table['undefined']}")
                continue
            design_points = "  ".join(f"b{k}={v['semif_margin']:+.3f}" for k, v in sorted(table.items()))
            print(f"  {rung:>14}: {design_points}")
