"""Measuring an experiment: requirement resolution, the design point loop, the contrast.

One resolution site for requirements (``unresolved``), then the loop that materialises each
design point's five values, measures it or records why it could not be, and pairs the results. The
parts of the design that are decisions rather than mechanics are stated where they happen --
materialising a level once per run, keying a score matrix on the design point's *context* rather
than its subset, and capturing importances where the scoring actually occurred.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .. import config, evaluate, features
from ..data import accessors, reporting, splits, subsets
from ..data.contract import (
    Dataset,
    Diagnostic,
    Ordering,
    Requirement,
    Undefined,
    is_undefined,
)
from ..model import rankers
from .declaration import (
    ARTIFACT_PREFIX,
    FACTOR_DATASET,
    FACTOR_FEATURES,
    FACTOR_MODEL,
    FACTOR_SPLIT,
    FACTOR_SUBSET,
    FACTORS,
    Binding,
    Controls,
    DesignPoint,
    Environment,
    Experiment,
)
from .report import DesignPointResult, RunReport

# --- availability ----------------------------------------------------------


def _artifact_path(reference: str, caches: Mapping[str, Path]) -> Path:
    """Resolve an ``artifact:`` reference, against the run's caches then the filesystem."""
    named = caches.get(reference)
    if named is not None:
        return Path(named)
    path = Path(reference)
    return path if path.is_absolute() else config.WORKSPACE / path



def unresolved(
    requirement: str,
    ds: Dataset | None,
    caches: Mapping[str, Path],
) -> Undefined | None:
    """Why ``requirement`` cannot be met for this design point, or ``None`` if it can.

    One resolution site, mirroring ``accessors.INPUTS`` being the one catalogue: the two
    spellings are ``artifact:<path>`` and a :class:`~rts.data.contract.Requirement` value, and
    anything else raises rather than quietly resolving to "absent", for the same reason
    ``has_capability("coverge")`` raises.
    """
    if requirement.startswith(ARTIFACT_PREFIX):
        reference = requirement[len(ARTIFACT_PREFIX) :]
        path = _artifact_path(reference, caches)
        if path.exists():
            return None
        return Undefined(
            requirement=requirement,
            note=f"required artifact {path} does not exist",
        )
    try:
        needed = Requirement(requirement)
    except ValueError:
        known = [r.value for r in Requirement]
        raise ValueError(
            f"unknown requirement {requirement!r}: expected {ARTIFACT_PREFIX}<path> or one "
            f"of {known}"
        ) from None
    if ds is None or needed not in ds.available_requirements():
        name = ds.name if ds is not None else "<no dataset>"
        return Undefined(
            requirement=needed.value,
            note=f"needs {needed.value}, which dataset {name!r} does not provide",
        )
    return None



def _model_requirement(ranker: rankers.Ranker, ds: Dataset, env: Environment) -> Undefined | None:
    requirements = getattr(ranker, "requirements", None)
    if requirements is None:
        return None
    for requirement in requirements():
        reason = unresolved(requirement, ds, env.caches)
        if reason is not None:
            return reason
    return None



def _build_values(
    experiment: Experiment,
    env: Environment,
    design_point: DesignPoint,
    cache: dict[tuple[str, str], Any],
) -> dict[str, Any] | Undefined:
    """Materialise the design point's five values, reusing anything already built this run.

    The dataset is built first because every other role reads it. Level values are cached by
    ``(role, level name)``, so a dataset is built once however many models sweep over it.
    """
    factors = design_point.factors_dict()
    values: dict[str, Any] = {}
    for role in FACTORS:
        key = (role, design_point.name(role))
        if key not in cache:
            level = experiment.factor_for(role).get(design_point.name(role))
            cache[key] = level.build(
                Binding(env=env, dataset=values.get(FACTOR_DATASET))
            )
        value = cache[key]
        if is_undefined(value):
            return value
        values[role] = value
    # Applicability can depend on an interaction between roles, so it is asked per design point, once
    # every value is in hand. This is the only place the design point's factors are visible.
    for role in FACTORS:
        level = experiment.factor_for(role).get(design_point.name(role))
        reason = level.check(Binding(env=env, dataset=values[FACTOR_DATASET], factors=factors))
        if reason is not None:
            return reason
    return values



def _measure(
    experiment: Experiment,
    env: Environment,
    design_point: DesignPoint,
    values: Mapping[str, Any],
    matrix: features.FeatureMatrix,
    audit: Sequence[Diagnostic],
    ranker: rankers.Ranker,
    bm25: np.ndarray,
    candidate_sets: np.ndarray,
    probes: Sequence[float],
    scores_cache: dict[tuple, np.ndarray],
    score_key: tuple,
) -> tuple[DesignPointResult | Undefined, dict[float, dict[int, bool]]]:
    """Measure one design point. Returns the result (or why it is undefined) and the probe hits."""
    ds: Dataset = values[FACTOR_DATASET]
    block: features.FeatureBlock = values[FACTOR_FEATURES]
    subset: subsets.Subset = values[FACTOR_SUBSET]
    split: splits.Split = values[FACTOR_SPLIT]

    unavailable_population = subset.unavailable(ds)
    if unavailable_population is not None:
        return unavailable_population, {}

    external = block.external_inputs
    if external:
        return (
            Undefined(
                requirement="feature.external_inputs",
                note=(
                    f"feature block {block.name!r} needs caller-supplied inputs "
                    f"{sorted(external)}, which a run does not supply"
                ),
            ),
            {},
        )

    ctx = rankers.Context(
        ds=ds,
        features=matrix,
        split=split,
        bm25=bm25,
        seed=env.controls.effective_model_seed,
    )
    if score_key in scores_cache:
        # A score matrix is a function of the *context*, and the context is (dataset, features,
        # model, split). Subset is deliberately not in it: a subset restricts which rows
        # a metric is averaged over, not which pairs get scored, so re-scoring per subset
        # would double the cost of every subset sweep to produce an identical matrix.
        scores = scores_cache[score_key]
        seconds = 0.0
        importances: dict = {}
    else:
        started = time.perf_counter()
        scores = ranker.scores(ctx)
        seconds = time.perf_counter() - started
        scores_cache[score_key] = scores
        # Captured here, on the design point that actually scored, rather than read back off the
        # ranker afterwards: the level is shared, so a later design point in another context would
        # have overwritten it.
        importances = dict(getattr(ranker, "importances_", None) or {})

    evaluation = evaluate.evaluate_rows(
        scores,
        ds,
        split.test_idx,
        budgets=env.controls.budgets,
        n_bootstrap=env.controls.n_bootstrap,
        seed=env.controls.seed,
        candidate_sets=candidate_sets,
        subset=subset,
        # The layer always evaluates exactly the split's window, so this guard holds by
        # construction -- which is the point: it is the boundary a direct-rows caller would
        # cross, and it is asserted rather than assumed.
        split=split,
    )
    if not evaluation.measured:
        return (evaluation.undefined or Undefined("subset", "undefined")), {}

    # Paired hits are computed over the *subset's* rows, not the whole evaluation window.
    # A contrast pairs two design points within one group, and the group names a subset; pairing
    # over the wider window would silently include changes the group's metric never averaged
    # over, which changes the delta and its interval.
    _, positions = evaluate.subset_rows(ds, split.test_idx, subset)
    if is_undefined(positions):
        return positions, {}
    # The subset's size in the window *before* the fault filter, which ``evaluation.n_rows``
    # no longer records. Reported alongside it so a renderer cannot present the averaged count
    # as the subset's size.
    subset_rows = subset.rows(ds, split.test_idx)
    if is_undefined(subset_rows):
        return subset_rows, {}
    hits = {
        probe: evaluate.per_change_hits(scores, ds, split.test_idx[positions], probe, candidate_sets)
        for probe in probes
    }
    collected = matrix.diagnostics
    result = DesignPointResult(
        design_point=design_point,
        ranker=ranker.name,
        subset=evaluation.subset,
        n_rows=evaluation.n_rows,
        n_changes=evaluation.n_changes,
        n_population_rows=int(len(subset_rows)),
        dataset_metadata=ds.metadata(),
        split={
            "fraction": split.fraction,
            "shuffle": split.shuffle,
            "seed": split.seed,
            "effective_order": split.effective_order.value,
        },
        features=matrix.audit(),
        # Two vocabularies, kept apart: ``diagnostics`` are the *derivation's* caveats (a block
        # whose temporal family was withheld, a temporal feature on a synthetic order), while
        # ``audit`` is what the dataset and split say about trusting a number at all. Merging
        # them loses the distinction a consumer acts on.
        diagnostics=tuple(w.to_dict() for w in collected),
        audit=tuple(w.to_dict() for w in audit),
        results=evaluate.results_to_dicts(evaluation.results or []),
        seconds=seconds,
        importances=importances,
    )
    return result, hits



def _compare(
    experiment: Experiment,
    env: Environment,
    results: Sequence[DesignPointResult],
    hits: Mapping[str, Mapping[float, dict[int, bool]]],
    unmeasured_keys: Mapping[str, Undefined],
) -> list[dict]:
    """Paired deltas, grouped by every role except the one the contrast varies."""
    out: list[dict] = []
    for contrast in experiment.contrasts:
        groups: dict[tuple[tuple[str, str], ...], dict[str, DesignPoint]] = defaultdict(dict)
        for result in results:
            group = tuple(
                (role, name) for role, name in result.design_point.factors if role != contrast.role
            )
            groups[group][result.design_point.name(contrast.role)] = result.design_point
        for group, at_role in groups.items():
            record = {
                "role": contrast.role,
                "reference": contrast.reference,
                "probe_budget": contrast.probe_budget,
                "group": dict(group),
            }
            reference_cell = at_role.get(contrast.reference)
            if reference_cell is None:
                out.append(
                    {
                        **record,
                        "design_point": None,
                        "measured": False,
                        "note": f"reference {contrast.reference!r} is not measured in this group",
                    }
                )
                continue
            for name, design_point in at_role.items():
                if name == contrast.reference:
                    continue
                if design_point.key not in hits:
                    reason = unmeasured_keys.get(design_point.key)
                    out.append(
                        {
                            **record,
                            "design_point": name,
                            "measured": False,
                            "note": reason.note if reason else "design_point was not measured",
                        }
                    )
                    continue
                stat = evaluate.paired_bootstrap(
                    hits[design_point.key][contrast.probe_budget],
                    hits[reference_cell.key][contrast.probe_budget],
                    env.controls.paired_resamples,
                    env.controls.seed,
                )
                out.append({**record, "design_point": name, "measured": True, **stat})
    return out



def run(
    experiment: Experiment,
    out_dir: Path | str | None = None,
    *,
    controls: Controls | None = None,
    shared: Mapping[str, Any] | None = None,
    caches: Mapping[str, Path] | None = None,
    tiers: Sequence[str] | None = None,
    scores: dict[tuple, np.ndarray] | None = None,
    save: bool = True,
    verbose: bool = True,
) -> RunReport:
    """Measure every design point of ``experiment`` and return the report.

    The parameters are the free choices a *run* makes as opposed to the ones the experiment
    declares: where to write, which controls to use if not the declared ones, what to inject, and
    which cost tiers to spend. All of them are recorded in the report.

    ``scores`` lets a caller share score matrices *between* runs, which matters when one
    experiment is split into two because a control differs -- the headline condition and the low-co-occurrence condition
    report different budget sets, and budgets are a control, so they cannot be one run. Reuse is
    sound because a score matrix is a function of the design point's context, which is what the key
    records; it is not a function of the averaging subset, which is why the low_cooccurrence corners
    cost nothing to add.
    """
    env = Environment(
        controls=controls or experiment.controls,
        out_dir=Path(out_dir) if out_dir is not None else config.ARTIFACTS,
        caches=dict(caches or {}),
        shared=dict(shared or {}),
    )
    started = time.perf_counter()
    enabled = set(tiers) if tiers is not None else None
    probes = experiment.probe_budgets()

    if verbose:
        print("=" * 78)
        print(f"experiment: {experiment.name}")
        print("=" * 78)
        for role, factor in experiment.factors().items():
            print(f"  {role:>11}: {', '.join(factor.names())}")
        print(f"  {'controls':>11}: {env.controls.to_dict()}")
        if enabled is not None:
            print(f"  {'tiers':>11}: {sorted(enabled)}")

    report = RunReport(
        experiment=experiment.name,
        note=experiment.note,
        environment=env,
        factors={role: factor.metadata() for role, factor in experiment.factors().items()},
        comparisons_declared=[
            {
                "role": c.role,
                "reference": c.reference,
                "probe_budget": c.probe_budget,
                "note": c.note,
            }
            for c in experiment.contrasts
        ],
    )

    built: dict[tuple[str, str], Any] = {}
    matrices: dict[tuple, features.FeatureMatrix] = {}
    audits: dict[tuple[str, str], tuple[Diagnostic, ...]] = {}
    score_cache: dict[tuple, np.ndarray] = {} if scores is None else scores
    bm25s: dict[str, np.ndarray] = {}
    candidate_masks: dict[str, np.ndarray] = {}
    hits: dict[str, dict[float, dict[int, bool]]] = {}
    unmeasured_keys: dict[str, Undefined] = {}

    for design_point in experiment.design_points():
        if enabled is not None and design_point.tier not in enabled:
            reason = Undefined(
                requirement=f"tier:{design_point.tier}",
                note=f"cost tier {design_point.tier!r} was not enabled for this run",
            )
            unmeasured_keys[design_point.key] = reason
            report.undefined.append(
                {**design_point.to_dict(), "measured": False, **reason.to_dict()}
            )
            continue

        values = _build_values(experiment, env, design_point, built)
        if is_undefined(values):
            unmeasured_keys[design_point.key] = values
            report.undefined.append(
                {**design_point.to_dict(), "measured": False, **values.to_dict()}
            )
            if verbose:
                print(f"\n[undefined] {design_point.key}\n  {values.note}")
            continue

        ds: Dataset = values[FACTOR_DATASET]
        model_reason = _model_requirement(values[FACTOR_MODEL], ds, env)
        if model_reason is not None:
            unmeasured_keys[design_point.key] = model_reason
            report.undefined.append(
                {**design_point.to_dict(), "measured": False, **model_reason.to_dict()}
            )
            if verbose:
                print(f"\n[undefined] {design_point.key}\n  {model_reason.note}")
            continue

        split: splits.Split = values[FACTOR_SPLIT]
        dataset_name = design_point.name(FACTOR_DATASET)
        features_name = design_point.name(FACTOR_FEATURES)
        # ``history`` is derived, not configured: the effective ordering of the run is the
        # dataset's unless the split shuffles, and it is the design point's split that decides. It is
        # part of the matrix key because a block with the temporal family withheld keeps the
        # same column list as one without.
        derived = split.effective_order is Ordering.NATURAL
        use_temporal = derived if experiment.temporal is None else experiment.temporal
        matrix_key = (dataset_name, features_name, design_point.name(FACTOR_SPLIT), use_temporal)
        if matrix_key not in matrices:
            matrices[matrix_key] = features.structured(
                ds, temporal=use_temporal, block=values[FACTOR_FEATURES]
            )
        matrix = matrices[matrix_key]

        if dataset_name not in bm25s:
            bm25s[dataset_name] = features.text.build_bm25_scores(ds)
        if dataset_name not in candidate_masks:
            candidate_masks[dataset_name] = accessors.candidate_sets(ds, env.controls.candidate_policy)
        audit_key = (dataset_name, design_point.name(FACTOR_SPLIT))
        if audit_key not in audits:
            audits[audit_key] = reporting.audit(ds, split)
            report.dataset_stats[f"{dataset_name}|{audit_key[1]}"] = {
                "metadata": ds.metadata(),
                "describe": reporting.describe(ds, split),
                # Recorded rather than recomputed by a renderer. ``None`` when the dataset
                # declares no coverage, because the statistic is not defined without it --
                # which is a fact about the data, not a reason to fail the run.
                "recurrence": (
                    reporting.recurrence(ds)
                    if ds.has_capability("coverage")
                    else None
                ),
            }

        result, cell_hits = _measure(
            experiment,
            env,
            design_point,
            values,
            matrix,
            audits[audit_key],
            values[FACTOR_MODEL],
            bm25s[dataset_name],
            candidate_masks[dataset_name],
            probes,
            score_cache,
            (
                dataset_name,
                ds.name,
                features_name,
                design_point.name(FACTOR_MODEL),
                design_point.name(FACTOR_SPLIT),
                round(split.fraction, 6),
                split.shuffle,
                split.seed,
                use_temporal,
                env.controls.candidate_policy,
                # The *model's* seed is part of the context, because it changes what the model
                # computes. Without it, re-fitting one model under several seeds would silently
                # reuse the first fit -- which is precisely the contrast a seed sweep is for.
                env.controls.effective_model_seed,
            ),
        )
        if is_undefined(result):
            unmeasured_keys[design_point.key] = result
            report.undefined.append(
                {**design_point.to_dict(), "measured": False, **result.to_dict()}
            )
            if verbose:
                print(f"\n[undefined] {design_point.key}\n  {result.note}")
            continue

        report.design_points.append(result)
        hits[design_point.key] = cell_hits
        # Every design point of one (dataset, subset) reports the same two counts, so first-seen
        # wins and the map stays a description of the subset rather than of a design point. The
        # key carries the dataset level as well, because the same subset name under two
        # datasets is two subsets -- see ``subset_sizes``.
        report.subset_sizes.setdefault(
            f"{design_point.name(FACTOR_DATASET)}|{result.subset}",
            {"changes": result.n_population_rows, "faults": result.n_rows},
        )
        if verbose:
            probe_note = ""
            if probes:
                probe = probes[0]
                row = next((r for r in result.results if r["budget"] == probe), None)
                if row is not None:
                    probe_note = f"  b{probe:.2f}={row['recall']:.3f}"
            print(f"  {result.design_point.key}  ({result.seconds:.1f}s){probe_note}")

    report.contrasts = _compare(experiment, env, report.design_points, hits, unmeasured_keys)
    report.seconds = time.perf_counter() - started

    if verbose:
        print("\n" + report.format_table())
        if report.contrasts:
            print("\nPaired deltas:")
            for record in report.contrasts:
                if not record.get("measured"):
                    continue
                print(
                    f"  {record['design_point']:>28} vs {record['reference']:<12} "
                    f"b{record['probe_budget']:.2f} delta {record['delta']:+.3f} "
                    f"[{record['lo']:+.3f}, {record['hi']:+.3f}] "
                    f"p={record['p_value']:.4f} n={record['n']}"
                )
        n_unmeasured = len(report.undefined)
        print(
            f"\n{len(report.design_points)} design_point(s) measured"
            + (f", {n_unmeasured} undefined" if n_unmeasured else "")
            + f"  [{report.seconds:.1f}s]"
        )

    if save:
        path = report.save()
        if verbose:
            print(f"[done] wrote {path}")
    return report
