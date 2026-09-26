"""The study's experiments, expressed as values: axes by role, and the named arms.

This is the config package the experiment layer was built for. It holds the study's *choices*
-- which datasets, which rungs, which selectors, which populations, which budgets -- and
nothing about how a sweep is executed.

It used to be one module of about 1150 lines holding three catalogues, so the variation arms
were not findable without reading all of it. It is now a package, and the public names are
re-exported here, so ``studies.study_arm()``, ``studies.RUNGS`` and ``studies.semif_margins``
read exactly as they did:

* :mod:`.axes` -- the role-by-role element builders (datasets, feature blocks, selector sets,
  populations, split);
* :mod:`.arms` -- the named arms and the runner that shares one score cache between them;
* :mod:`.ladder` -- the traceability ladder's declarations and its arm;
* :mod:`.variations` -- the variation study's arms;
* :mod:`.readings` -- quantities computed *from* a report rather than recorded by the kernel.

**Axis builders are functions, not module constants**, wherever the elements hold state. An
:class:`~rts.experiment.Experiment` is immutable and could be a constant, but its axes hold
selector instances that train on use, so a fresh axis per arm keeps two runs from sharing one
model object.

**Two arms here are verification vehicles.** ``study_arm()``/``sparse_arm()`` reproduce
``artifacts/results_{full,covered}.json`` and ``ladder_arm()`` reproduces
``artifacts/ladder.json``, both through ``scripts/verify_experiment_layer.py``. The drivers that
render those artifacts -- ``rts/pipeline.py`` and ``rts/ladder.py`` -- are thin: they run an arm
declared here and write it in the shape the recorded numbers are written against. The remaining
drivers (``variations``, ``bundles``, ``bugsinpy``) still hold their own sweeps and are the next
migration (``docs/experiment.md`` §13).

Usage::

    python -m rts.studies study
    python -m rts.studies ladder.mutmut --tiers cpu
"""

from __future__ import annotations

import argparse

from ..experiment import run
from . import bugsinpy
from .arms import (
    SPARSE_BUDGETS,
    SPARSE_THRESHOLDS,
    dataset,
    run_study,
    semif_production_arm,
    sparse_arm,
    study_arm,
)
from .bugsinpy import (
    BUGSINPY_BUDGETS,
    bugsinpy_arm,
    run_bugsinpy,
)
from .axes import (
    ABLATION_MODELS,
    dataset_axis,
    model_axis,
    population_axis,
    semif_scoring_model_axis,
    split_axis,
    structured_feature_axis,
)
from .ladder import (
    LADDER_BUDGETS,
    LADDER_PROBE,
    LADDER_RESAMPLES,
    LADDER_TABLE_RESAMPLES,
    RUNGS,
    SEMIF_LADDER_CACHE,
    ladder_arm,
    ladder_feature_axis,
    ladder_model_axis,
    ladder_population_axis,
    ladder_populations,
    ladder_selectors,
    rung_block,
)
from .readings import semif_margins
from .variations import (
    SEED_SWEEP,
    STARVED_THRESHOLDS,
    VARIATION_BUDGETS,
    VARIATION_PROBE,
    VARIATION_RESAMPLES,
    VARIATION_TABLE_RESAMPLES,
    embed_arm,
    embed_cache,
    embed_model_axis,
    instruction_arm,
    instruction_cache,
    instruction_model_axis,
    instruction_names,
    redundancy_arm,
    redundancy_model_axis,
    starved_arm,
    starved_cache,
    starved_model_axis,
    starved_seeds_arm,
    variation_comparisons,
)

__all__ = [
    "ABLATION_MODELS",
    "ARMS",
    "BUGSINPY_BUDGETS",
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
    "bugsinpy_arm",
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
    "run_bugsinpy",
    "run_study",
    "semif_margins",
    "semif_production_arm",
    "semif_scoring_model_axis",
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


# --- CLI --------------------------------------------------------------------

ARMS: dict[str, object] = {
    "study": study_arm,
    "study.covered": lambda: study_arm(candidates="covered"),
    "ladder.mutmut": lambda: ladder_arm("mutmut"),
    "ladder.full": lambda: ladder_arm("full"),
    # Declared so score *production* is addressable like any other arm: the GPU tier is what
    # makes it score rather than report an unmeasured cell.
    "semif.produce": lambda: semif_production_arm(),
    # One budget, which is the arm's unit: the recorded intervals are per-budget, so the full
    # sweep is four runs through ``studies.run_bugsinpy`` (see ``rts/studies/bugsinpy``).
    "bugsinpy": lambda: bugsinpy_arm(0.05),
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
