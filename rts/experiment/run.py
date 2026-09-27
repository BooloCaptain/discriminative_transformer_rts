"""Measuring an experiment: requirement resolution, the cell loop, the comparison.

One resolution site for requirements (``unresolved``), then the loop that materialises each
cell's five values, measures it or records why it could not be, and pairs the results. The
parts of the design that are decisions rather than mechanics are stated where they happen --
materialising an element once per run, keying a score matrix on the cell's *context* rather
than its population, and capturing importances where the scoring actually occurred.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .. import config, evaluate, features
from ..data import accessors, populations, reporting, splits
from ..data.contract import Dataset, Ordering, Requirement, Unmeasured, is_unmeasured
from ..model import selectors
from .declaration import (
    ARTIFACT_PREFIX,
    ROLE_DATASET,
    ROLE_FEATURES,
    ROLE_MODEL,
    ROLE_POPULATION,
    ROLE_SPLIT,
    ROLES,
    Binding,
    Cell,
    Environment,
    Experiment,
    Knobs,
)
from .report import CellResult, RunReport

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
) -> Unmeasured | None:
    """Why ``requirement`` cannot be met for this cell, or ``None`` if it can.

    One resolution site, mirroring ``accessors.MATERIAL`` being the one catalogue: the two
    spellings are ``artifact:<path>`` and a :class:`~rts.data.contract.Requirement` value, and
    anything else raises rather than quietly resolving to "absent", for the same reason
    ``has_capability("coverge")`` raises.
    """
    if requirement.startswith(ARTIFACT_PREFIX):
        reference = requirement[len(ARTIFACT_PREFIX) :]
        path = _artifact_path(reference, caches)
        if path.exists():
            return None
        return Unmeasured(
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
        return Unmeasured(
            requirement=needed.value,
            note=f"needs {needed.value}, which dataset {name!r} does not provide",
        )
    return None



def _model_requirement(selector: selectors.Selector, ds: Dataset, env: Environment) -> Unmeasured | None:
    requirements = getattr(selector, "requirements", None)
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
    cell: Cell,
    cache: dict[tuple[str, str], Any],
) -> dict[str, Any] | Unmeasured:
    """Materialise the cell's five values, reusing anything already built this run.

    The dataset is built first because every other role reads it. Element values are cached by
    ``(role, element name)``, so a dataset is built once however many models sweep over it.
    """
    factors = cell.factors_dict()
    values: dict[str, Any] = {}
    for role in ROLES:
        key = (role, cell.name(role))
        if key not in cache:
            element = experiment.axis_for(role).get(cell.name(role))
            cache[key] = element.build(
                Binding(env=env, dataset=values.get(ROLE_DATASET))
            )
        value = cache[key]
        if is_unmeasured(value):
            return value
        values[role] = value
    # Applicability can depend on an interaction between roles, so it is asked per cell, once
    # every value is in hand. This is the only place the cell's factors are visible.
    for role in ROLES:
        element = experiment.axis_for(role).get(cell.name(role))
        reason = element.check(Binding(env=env, dataset=values[ROLE_DATASET], factors=factors))
        if reason is not None:
            return reason
    return values



def _measure(
    experiment: Experiment,
    env: Environment,
    cell: Cell,
    values: Mapping[str, Any],
    matrix: features.FeatureMatrix,
    audit: Sequence[Warning],
    selector: selectors.Selector,
    bm25: np.ndarray,
    candidates: np.ndarray,
    probes: Sequence[float],
    scores_cache: dict[tuple, np.ndarray],
    score_key: tuple,
) -> tuple[CellResult | Unmeasured, dict[float, dict[int, bool]]]:
    """Measure one cell. Returns the result (or why it is unmeasured) and the probe hits."""
    ds: Dataset = values[ROLE_DATASET]
    block: features.FeatureBlock = values[ROLE_FEATURES]
    population: populations.Population = values[ROLE_POPULATION]
    split: splits.Split = values[ROLE_SPLIT]

    unavailable_population = population.unavailable(ds)
    if unavailable_population is not None:
        return unavailable_population, {}

    external = block.external_material
    if external:
        return (
            Unmeasured(
                requirement="feature.external_material",
                note=(
                    f"feature block {block.name!r} needs caller-supplied material "
                    f"{sorted(external)}, which a run does not supply"
                ),
            ),
            {},
        )

    ctx = selectors.Context(
        ds=ds,
        features=matrix,
        split=split,
        bm25=bm25,
        seed=env.knobs.effective_model_seed,
    )
    if score_key in scores_cache:
        # A score matrix is a function of the *context*, and the context is (dataset, features,
        # model, split). Population is deliberately not in it: a population restricts which rows
        # a metric is averaged over, not which pairs get scored, so re-scoring per population
        # would double the cost of every population sweep to produce an identical matrix.
        scores = scores_cache[score_key]
        seconds = 0.0
        importances: dict = {}
    else:
        started = time.perf_counter()
        scores = selector.scores(ctx)
        seconds = time.perf_counter() - started
        scores_cache[score_key] = scores
        # Captured here, on the cell that actually scored, rather than read back off the
        # selector afterwards: the element is shared, so a later cell in another context would
        # have overwritten it.
        importances = dict(getattr(selector, "importances_", None) or {})

    evaluation = evaluate.evaluate_rows(
        scores,
        ds,
        split.test_idx,
        budgets=env.knobs.budgets,
        n_bootstrap=env.knobs.n_bootstrap,
        seed=env.knobs.seed,
        candidates=candidates,
        population=population,
        # The layer always evaluates exactly the split's window, so this guard holds by
        # construction -- which is the point: it is the boundary a direct-rows caller would
        # cross, and it is asserted rather than assumed.
        split=split,
    )
    if not evaluation.measured:
        return (evaluation.unmeasured or Unmeasured("population", "unmeasured")), {}

    # Paired hits are computed over the *population's* rows, not the whole evaluation window.
    # A comparison pairs two cells within one group, and the group names a population; pairing
    # over the wider window would silently include changes the group's metric never averaged
    # over, which changes the delta and its interval.
    _, positions = evaluate.population_rows(ds, split.test_idx, population)
    if is_unmeasured(positions):
        return positions, {}
    # The population's size in the window *before* the fault filter, which ``evaluation.n_rows``
    # no longer records. Reported alongside it so a renderer cannot present the averaged count
    # as the population's size.
    population_rows = population.rows(ds, split.test_idx)
    if is_unmeasured(population_rows):
        return population_rows, {}
    hits = {
        probe: evaluate.per_change_hits(scores, ds, split.test_idx[positions], probe, candidates)
        for probe in probes
    }
    collected = matrix.warnings
    result = CellResult(
        cell=cell,
        selector=selector.name,
        population=evaluation.population,
        n_rows=evaluation.n_rows,
        n_changes=evaluation.n_changes,
        n_population_rows=int(len(population_rows)),
        dataset_declaration=ds.declaration(),
        split={
            "fraction": split.fraction,
            "shuffle": split.shuffle,
            "seed": split.seed,
            "effective_ordering": split.effective_ordering.value,
        },
        features=matrix.audit(),
        # Two vocabularies, kept apart: ``warnings`` are the *derivation's* caveats (a block
        # whose history family was withheld, a history feature on an imposed order), while
        # ``audit`` is what the dataset and split say about trusting a number at all. Merging
        # them loses the distinction a consumer acts on.
        warnings=tuple(w.to_dict() for w in collected),
        audit=tuple(w.to_dict() for w in audit),
        results=evaluate.results_to_dicts(evaluation.results or []),
        seconds=seconds,
        importances=importances,
    )
    return result, hits



def _compare(
    experiment: Experiment,
    env: Environment,
    results: Sequence[CellResult],
    hits: Mapping[str, Mapping[float, dict[int, bool]]],
    unmeasured_keys: Mapping[str, Unmeasured],
) -> list[dict]:
    """Paired deltas, grouped by every role except the one the comparison varies."""
    out: list[dict] = []
    for comparison in experiment.comparisons:
        groups: dict[tuple[tuple[str, str], ...], dict[str, Cell]] = defaultdict(dict)
        for result in results:
            group = tuple(
                (role, name) for role, name in result.cell.factors if role != comparison.role
            )
            groups[group][result.cell.name(comparison.role)] = result.cell
        for group, at_role in groups.items():
            record = {
                "role": comparison.role,
                "reference": comparison.reference,
                "probe_budget": comparison.probe_budget,
                "group": dict(group),
            }
            reference_cell = at_role.get(comparison.reference)
            if reference_cell is None:
                out.append(
                    {
                        **record,
                        "cell": None,
                        "measured": False,
                        "note": f"reference {comparison.reference!r} is not measured in this group",
                    }
                )
                continue
            for name, cell in at_role.items():
                if name == comparison.reference:
                    continue
                if cell.key not in hits:
                    reason = unmeasured_keys.get(cell.key)
                    out.append(
                        {
                            **record,
                            "cell": name,
                            "measured": False,
                            "note": reason.note if reason else "cell was not measured",
                        }
                    )
                    continue
                stat = evaluate.paired_bootstrap(
                    hits[cell.key][comparison.probe_budget],
                    hits[reference_cell.key][comparison.probe_budget],
                    env.knobs.paired_resamples,
                    env.knobs.seed,
                )
                out.append({**record, "cell": name, "measured": True, **stat})
    return out



def run(
    experiment: Experiment,
    out_dir: Path | str | None = None,
    *,
    knobs: Knobs | None = None,
    shared: Mapping[str, Any] | None = None,
    caches: Mapping[str, Path] | None = None,
    tiers: Sequence[str] | None = None,
    scores: dict[tuple, np.ndarray] | None = None,
    save: bool = True,
    verbose: bool = True,
) -> RunReport:
    """Measure every cell of ``experiment`` and return the report.

    The parameters are the free choices a *run* makes as opposed to the ones the experiment
    declares: where to write, which knobs to use if not the declared ones, what to inject, and
    which cost tiers to spend. All of them are recorded in the report.

    ``scores`` lets a caller share score matrices *between* runs, which matters when one
    experiment is split into two because a knob differs -- the headline arm and the sparse arm
    report different budget sets, and budgets are a knob, so they cannot be one run. Reuse is
    sound because a score matrix is a function of the cell's context, which is what the key
    records; it is not a function of the averaging population, which is why the sparse corners
    cost nothing to add.
    """
    env = Environment(
        knobs=knobs or experiment.knobs,
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
        for role, axis in experiment.axes().items():
            print(f"  {role:>11}: {', '.join(axis.names())}")
        print(f"  {'knobs':>11}: {env.knobs.to_dict()}")
        if enabled is not None:
            print(f"  {'tiers':>11}: {sorted(enabled)}")

    report = RunReport(
        experiment=experiment.name,
        note=experiment.note,
        environment=env,
        axes={role: axis.declaration() for role, axis in experiment.axes().items()},
        comparisons_declared=[
            {
                "role": c.role,
                "reference": c.reference,
                "probe_budget": c.probe_budget,
                "note": c.note,
            }
            for c in experiment.comparisons
        ],
    )

    built: dict[tuple[str, str], Any] = {}
    matrices: dict[tuple, features.FeatureMatrix] = {}
    audits: dict[tuple[str, str], tuple[Warning, ...]] = {}
    score_cache: dict[tuple, np.ndarray] = {} if scores is None else scores
    bm25s: dict[str, np.ndarray] = {}
    candidate_masks: dict[str, np.ndarray] = {}
    hits: dict[str, dict[float, dict[int, bool]]] = {}
    unmeasured_keys: dict[str, Unmeasured] = {}

    for cell in experiment.cells():
        if enabled is not None and cell.tier not in enabled:
            reason = Unmeasured(
                requirement=f"tier:{cell.tier}",
                note=f"cost tier {cell.tier!r} was not enabled for this run",
            )
            unmeasured_keys[cell.key] = reason
            report.unmeasured.append(
                {**cell.to_dict(), "measured": False, **reason.to_dict()}
            )
            continue

        values = _build_values(experiment, env, cell, built)
        if is_unmeasured(values):
            unmeasured_keys[cell.key] = values
            report.unmeasured.append(
                {**cell.to_dict(), "measured": False, **values.to_dict()}
            )
            if verbose:
                print(f"\n[unmeasured] {cell.key}\n  {values.note}")
            continue

        ds: Dataset = values[ROLE_DATASET]
        model_reason = _model_requirement(values[ROLE_MODEL], ds, env)
        if model_reason is not None:
            unmeasured_keys[cell.key] = model_reason
            report.unmeasured.append(
                {**cell.to_dict(), "measured": False, **model_reason.to_dict()}
            )
            if verbose:
                print(f"\n[unmeasured] {cell.key}\n  {model_reason.note}")
            continue

        split: splits.Split = values[ROLE_SPLIT]
        dataset_name = cell.name(ROLE_DATASET)
        features_name = cell.name(ROLE_FEATURES)
        # ``history`` is derived, not configured: the effective ordering of the run is the
        # dataset's unless the split shuffles, and it is the cell's split that decides. It is
        # part of the matrix key because a block with the history family withheld keeps the
        # same column list as one without.
        derived = split.effective_ordering is Ordering.OBSERVED
        use_history = derived if experiment.history is None else experiment.history
        matrix_key = (dataset_name, features_name, cell.name(ROLE_SPLIT), use_history)
        if matrix_key not in matrices:
            matrices[matrix_key] = features.structured(
                ds, history=use_history, block=values[ROLE_FEATURES]
            )
        matrix = matrices[matrix_key]

        if dataset_name not in bm25s:
            bm25s[dataset_name] = features.text.build_bm25_scores(ds)
        if dataset_name not in candidate_masks:
            candidate_masks[dataset_name] = accessors.candidates(ds, env.knobs.candidates)
        audit_key = (dataset_name, cell.name(ROLE_SPLIT))
        if audit_key not in audits:
            audits[audit_key] = reporting.audit(ds, split)
            report.dataset_stats[f"{dataset_name}|{audit_key[1]}"] = {
                "declaration": ds.declaration(),
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
            cell,
            values,
            matrix,
            audits[audit_key],
            values[ROLE_MODEL],
            bm25s[dataset_name],
            candidate_masks[dataset_name],
            probes,
            score_cache,
            (
                dataset_name,
                ds.name,
                features_name,
                cell.name(ROLE_MODEL),
                cell.name(ROLE_SPLIT),
                round(split.fraction, 6),
                split.shuffle,
                split.seed,
                use_history,
                env.knobs.candidates,
                # The *model's* seed is part of the context, because it changes what the model
                # computes. Without it, re-fitting one model under several seeds would silently
                # reuse the first fit -- which is precisely the comparison a seed sweep is for.
                env.knobs.effective_model_seed,
            ),
        )
        if is_unmeasured(result):
            unmeasured_keys[cell.key] = result
            report.unmeasured.append(
                {**cell.to_dict(), "measured": False, **result.to_dict()}
            )
            if verbose:
                print(f"\n[unmeasured] {cell.key}\n  {result.note}")
            continue

        report.cells.append(result)
        hits[cell.key] = cell_hits
        # Every cell of one (dataset, population) reports the same two counts, so first-seen
        # wins and the map stays a description of the population rather than of a cell. The
        # key carries the dataset element as well, because the same population name under two
        # datasets is two populations -- see ``population_sizes``.
        report.population_sizes.setdefault(
            f"{cell.name(ROLE_DATASET)}|{result.population}",
            {"changes": result.n_population_rows, "faults": result.n_rows},
        )
        if verbose:
            probe_note = ""
            if probes:
                probe = probes[0]
                row = next((r for r in result.results if r["budget"] == probe), None)
                if row is not None:
                    probe_note = f"  b{probe:.2f}={row['recall']:.3f}"
            print(f"  {result.cell.key}  ({result.seconds:.1f}s){probe_note}")

    report.comparisons = _compare(experiment, env, report.cells, hits, unmeasured_keys)
    report.seconds = time.perf_counter() - started

    if verbose:
        print("\n" + report.format_table())
        if report.comparisons:
            print("\nPaired deltas:")
            for record in report.comparisons:
                if not record.get("measured"):
                    continue
                print(
                    f"  {record['cell']:>28} vs {record['reference']:<12} "
                    f"b{record['probe_budget']:.2f} delta {record['delta']:+.3f} "
                    f"[{record['lo']:+.3f}, {record['hi']:+.3f}] "
                    f"p={record['p_value']:.4f} n={record['n']}"
                )
        n_unmeasured = len(report.unmeasured)
        print(
            f"\n{len(report.cells)} cell(s) measured"
            + (f", {n_unmeasured} unmeasured" if n_unmeasured else "")
            + f"  [{report.seconds:.1f}s]"
        )

    if save:
        path = report.save()
        if verbose:
            print(f"[done] wrote {path}")
    return report
