"""The experiment layer: a declared sweep over five roles, and the run that measures it.

The harness already had three of the five things an experiment sweeps over as plain values
with declarations -- :class:`~rts.contract.Dataset`, ``FeatureBlock`` and
:class:`~rts.populations.Population` -- and it had no layer that composed them. Instead each
study arm was a hand-written driver that rebuilt the same sequence and kept the study's choices
as its own module constants. The sweep existed as control flow rather than as data, and "a named
variant of one input" was implemented three times (the ladder's rungs, bundle rungs, and the
instruction variants). Two of those drivers are now renderers over a declared arm
(``rts/pipeline.py``, ``rts/ladder.py``); the rest are listed in ``experiment.md`` §13.

This module is that missing layer. See ``experiment.md`` for the design and its rationale.

**Five roles.** An experiment sweeps ``dataset``, ``features``, ``model``, ``population`` and
``split``. A new *element* in any role is cheap -- that is what most new experimental
dimensions are. A new *role* is a deliberate kernel change, because a role has to say what it
feeds. Every factor the study has conceived already reduces to one of the five: a rung is a
feature block with families withheld, an instruction variant is a model reading a different
cache, a bundle rung is a dataset derived from a base, a starvation threshold is a population,
and a label source is a dataset.

**Cells, not a product.** A cell is one point in the product, and it is either measured or
carries an :class:`~rts.contract.Unmeasured` naming what stopped it. Unmeasured cells are
*reported*, never dropped: dropping is what turns "we asked and could not answer" into a
silently halved comparison, which is the failure mode ``refactor.md`` §6 exists to prevent.

**Two authorities.** The *element* declares its name, cost tier and estimated seconds -- facts
about this run's use of a thing. The *value* declares its requirements (``Selector.requirements()``,
``FeatureGroup.needs``, ``Population.needs``, ``Dataset.capabilities()``) -- facts about the
thing itself, written where the material is read so they cannot drift from it.

**Elements are materialised once per run** and shared across the cells that use them, which is
what makes a 48-cell grid affordable: the dataset is built once. This is sound because these
are values (``refactor.md`` §2). The one consequence is that a stateful selector's post-hoc
attributes describe only its most recent call, so importances are captured at the point of
scoring rather than read back afterwards.
"""

from __future__ import annotations

import itertools
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import (
    accessors,
    config,
    evaluate,
    features,
    models,
    populations,
    reporting,
    splits,
)
from .contract import (
    Dataset,
    Ordering,
    Requirement,
    Unmeasured,
    Warning,
    is_unmeasured,
)

__all__ = [
    "ARTIFACT_PREFIX",
    "ROLES",
    "ROLE_DATASET",
    "ROLE_FEATURES",
    "ROLE_MODEL",
    "ROLE_POPULATION",
    "ROLE_SPLIT",
    "ROW_CHANGING_ROLES",
    "Axis",
    "Binding",
    "Cell",
    "CellResult",
    "Comparison",
    "Element",
    "Environment",
    "Experiment",
    "Knobs",
    "RunReport",
    "constant",
    "run",
    "unavailable",
]

ROLE_DATASET = "dataset"
ROLE_FEATURES = "features"
ROLE_MODEL = "model"
ROLE_POPULATION = "population"
ROLE_SPLIT = "split"

#: The roles an experiment sweeps, in the order a cell's key names them.
ROLES: tuple[str, ...] = (ROLE_DATASET, ROLE_FEATURES, ROLE_MODEL, ROLE_POPULATION, ROLE_SPLIT)

#: Roles whose variation changes *which rows exist* or which pairs are rankable. Two cells
#: differing in one of these are not two measurements of one quantity, so a paired
#: comparison across them is refused rather than reported (``experiment.md`` §7).
ROW_CHANGING_ROLES = frozenset({ROLE_DATASET, ROLE_POPULATION, ROLE_SPLIT})

ARTIFACT_PREFIX = "artifact:"


# --- knobs and environment -------------------------------------------------


@dataclass(frozen=True)
class Knobs:
    """Run-level constants: a choice that is free, not one the harness can derive.

    These do not multiply into cells. They parameterise every cell and are recorded once, so
    a difference in a result can never be attributed to a knob that moved silently.
    """

    seed: int = config.SEED
    budgets: tuple[float, ...] = config.DEFAULT_BUDGETS
    n_bootstrap: int = config.DEFAULT_BOOTSTRAP
    candidates: str = "full"
    #: Resamples for a paired test, when it differs from the table CI's. Two quantities, two
    #: precision needs: the recorded ladder used 1000 for its table intervals and 2000 for its
    #: paired p-values. ``None`` means "the same as ``n_bootstrap``".
    n_bootstrap_paired: int | None = None
    #: Seed for the *model's* randomness, when that should differ from the run's. ``seed`` fixes
    #: the imposed change order and the temporal split, and a score cache is keyed to that order
    #: -- so a study that wants to re-fit one model under several seeds must vary this one and
    #: hold ``seed``, or it would move the data underneath the cache and invalidate the paired
    #: comparison instead of testing it. ``None`` means "the same as ``seed``".
    model_seed: int | None = None

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "model_seed": self.model_seed,
            "budgets": list(self.budgets),
            "n_bootstrap": self.n_bootstrap,
            "n_bootstrap_paired": self.paired_resamples,
            "candidates": self.candidates,
        }

    @property
    def paired_resamples(self) -> int:
        return self.n_bootstrap if self.n_bootstrap_paired is None else self.n_bootstrap_paired

    @property
    def effective_model_seed(self) -> int:
        """What a model's randomness is seeded with: its own seed, else the run's."""
        return self.seed if self.model_seed is None else self.model_seed


@dataclass(frozen=True)
class Environment:
    """What a run offers to the elements it builds.

    ``caches`` maps a short name to an artifact path, so a run can point at a different cache
    without editing the experiment that names the requirement. ``shared`` is free-form and
    exists so a *caller* can inject an object an element expects -- a stub dataset in a test,
    or one expensive base dataset shared by several derived elements.
    """

    knobs: Knobs = Knobs()
    out_dir: Path = config.ARTIFACTS
    caches: Mapping[str, Path] = field(default_factory=dict)
    shared: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "knobs": self.knobs.to_dict(),
            "out_dir": str(self.out_dir),
            "caches": {k: str(v) for k, v in self.caches.items()},
            "shared": sorted(self.shared),
        }


@dataclass(frozen=True)
class Binding:
    """What an :class:`Element` is handed when it is built.

    ``dataset`` is the cell's resolved dataset for every role except ``dataset`` itself, which
    is built first and sees ``None``. A dataset element that derives from another dataset
    simply constructs or shares its base inside ``make`` -- no extra machinery is needed,
    because ``make`` is already arbitrary Python.

    ``factors`` is the cell's factor names by role, and is populated **only when an element's
    applicability is checked**. It is empty while a value is being built, because a value is
    built once per run and shared across every cell that selects it: an element whose ``make``
    read this would silently be reused for cells it does not describe. Anything that genuinely
    varies by cell belongs in ``Element.applies``, which is asked per cell.
    """

    env: Environment
    dataset: Dataset | None = None
    factors: Mapping[str, str] = field(default_factory=dict)

    @property
    def knobs(self) -> Knobs:
        return self.env.knobs

    @property
    def caches(self) -> Mapping[str, Path]:
        return self.env.caches

    @property
    def shared(self) -> Mapping[str, Any]:
        return self.env.shared


# --- elements and axes -----------------------------------------------------


@dataclass(frozen=True)
class Element:
    """One variant of one input: a name, a builder, and what it costs to measure.

    ``applies`` is for the case where availability cannot be asked of one role's value alone.
    Normally a requirement is a property of a value (``Selector.requirements()``,
    ``Population.needs``), and the kernel resolves it against the dataset. But some variants
    are only meaningful in combination with a particular element of *another* role -- a score
    artifact that covers one population and not another -- and that is a fact the study config
    knows and no single value does. Declaring it here keeps the knowledge beside the variant
    rather than putting a special case in the kernel.
    """

    name: str
    make: Callable[[Binding], Any]
    tier: str = "cpu"
    estimated_seconds: float | None = None
    note: str = ""
    applies: Callable[[Binding], Unmeasured | None] | None = None

    def build(self, binding: Binding) -> Any:
        """The value this element denotes, or an :class:`Unmeasured` if it cannot exist."""
        return self.make(binding)

    def check(self, binding: Binding) -> Unmeasured | None:
        """Why this element does not apply to ``binding``'s cell, or ``None`` if it does."""
        if self.applies is None:
            return None
        return self.applies(binding)

    def declaration(self) -> dict:
        return {
            "name": self.name,
            "tier": self.tier,
            "estimated_seconds": self.estimated_seconds,
            "note": self.note,
            "applies": self.applies is not None,
        }


def constant(name: str, obj: Any, **kwargs: Any) -> Element:
    """An element whose value does not depend on the run.

    The value is shared by every cell that selects this element, which is intended: for a
    dataset it means the artifact is read once however many models are swept over it.
    """
    return Element(name=name, make=lambda _binding: obj, **kwargs)


def unavailable(name: str, reason: Unmeasured, **kwargs: Any) -> Element:
    """An element that can never be materialised, carrying why.

    Used by :meth:`Axis.map` to keep an element whose shared option cannot be expressed for
    it: the cell is then reported unmeasured rather than silently absent.
    """
    return Element(
        name=name,
        make=lambda _binding, reason=reason: reason,
        note=kwargs.pop("note", "") or reason.note,
        **kwargs,
    )


@dataclass(frozen=True)
class Axis:
    """A named, ordered set of variants of one role's input."""

    role: str
    elements: tuple[Element, ...]
    note: str = ""

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"unknown role {self.role!r}; known roles: {list(ROLES)}")
        names = [e.name for e in self.elements]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"axis {self.role!r} has duplicate element names: {duplicates}")
        if not self.elements:
            raise ValueError(f"axis {self.role!r} has no elements")

    def names(self) -> tuple[str, ...]:
        return tuple(e.name for e in self.elements)

    def get(self, name: str) -> Element:
        for element in self.elements:
            if element.name == name:
                return element
        raise KeyError(f"axis {self.role!r} has no element {name!r}; have {list(self.names())}")

    def has(self, name: str) -> bool:
        return any(e.name == name for e in self.elements)

    def select(self, *names: str) -> "Axis":
        """This axis restricted to ``names``, in the axis's own order."""
        wanted = set(names)
        unknown = wanted - set(self.names())
        if unknown:
            raise KeyError(f"axis {self.role!r} has no element(s) {sorted(unknown)}")
        return Axis(self.role, tuple(e for e in self.elements if e.name in wanted), self.note)

    def without(self, *names: str) -> "Axis":
        """This axis with ``names`` removed."""
        dropped = set(names)
        unknown = dropped - set(self.names())
        if unknown:
            raise KeyError(f"axis {self.role!r} has no element(s) {sorted(unknown)}")
        return Axis(self.role, tuple(e for e in self.elements if e.name not in dropped), self.note)

    def extend(self, *elements: Element) -> "Axis":
        return Axis(self.role, self.elements + tuple(elements), self.note)

    def map(self, fn: Callable[[Element], Element | Unmeasured]) -> "Axis":
        """Apply a shared dimension-level option to every element.

        ``fn`` returns the element as it should be with the option applied, or an
        :class:`Unmeasured` when the option cannot be expressed for that element -- which is a
        real case, not a degenerate one: "exclude the history family" is expressible for a
        column-reading model and meaningless for one that reads text pairs. An inapplicable
        element is kept as a poisoned element rather than dropped, so its cells appear as
        unmeasured findings.
        """
        out: list[Element] = []
        for element in self.elements:
            result = fn(element)
            if is_unmeasured(result):
                out.append(
                    unavailable(element.name, result, tier=element.tier, note=element.note)
                )
            else:
                out.append(result)
        return Axis(self.role, tuple(out), self.note)

    def declaration(self) -> list[dict]:
        return [e.declaration() for e in self.elements]


# --- cells and comparisons -------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One point in the product: one element per role, plus the run's knobs."""

    factors: tuple[tuple[str, str], ...]
    tier: str = "cpu"
    estimated_seconds: float | None = None

    def name(self, role: str) -> str:
        for r, n in self.factors:
            if r == role:
                return n
        raise KeyError(f"cell has no factor for role {role!r}")

    @property
    def key(self) -> str:
        """Stable identity of the cell: the element names, in role order."""
        return "|".join(f"{r}={n}" for r, n in self.factors)

    @property
    def label(self) -> str:
        return ", ".join(n for _, n in self.factors)

    def factors_dict(self) -> dict[str, str]:
        return dict(self.factors)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "factors": self.factors_dict(),
            "tier": self.tier,
            "estimated_seconds": self.estimated_seconds,
        }


@dataclass(frozen=True)
class Comparison:
    """A paired delta against a reference element of one role, at a probe budget.

    Only a *pairable* role may be named. Pairing is a claim that the two cells measure the same
    quantity on the same rows, which is false the moment the varying role changes the rows --
    so a row-changing role is refused at construction rather than producing a plausible number
    from two different populations.
    """

    role: str
    reference: str
    probe_budget: float
    note: str = ""

    def __post_init__(self) -> None:
        if self.role in ROW_CHANGING_ROLES:
            pairable = sorted(set(ROLES) - ROW_CHANGING_ROLES)
            raise ValueError(
                f"comparison over role {self.role!r} would pair different populations: "
                f"that role changes which rows exist, so only {pairable} give a paired delta"
            )


# --- the experiment --------------------------------------------------------


@dataclass(frozen=True)
class Experiment:
    """A declaration of what to sweep. A value: building one runs nothing.

    ``history`` is an **override**, not an option. Whether the cumulative history columns are
    present is derived from the dataset's ``ordering()`` and from whether the split shuffles
    (``splits.Split.effective_ordering``); the harness warns when a caller overrides it. It is
    exposed because the study's recorded arm does override it deliberately, and the warning
    travels with the cell.
    """

    name: str
    datasets: Axis
    features: Axis
    models: Axis
    populations: Axis
    splits: Axis
    knobs: Knobs = Knobs()
    comparisons: tuple[Comparison, ...] = ()
    history: bool | None = None
    tier_order: tuple[str, ...] = ("cpu", "gpu")
    note: str = ""

    def __post_init__(self) -> None:
        for role, axis in self.axes().items():
            if axis.role != role:
                raise ValueError(
                    f"experiment {self.name!r}: axis for {role!r} declares role {axis.role!r}"
                )
        if not self.tier_order:
            raise ValueError(f"experiment {self.name!r}: tier_order is empty")
        for comparison in self.comparisons:
            axis = self.axes()[comparison.role]
            if not axis.has(comparison.reference):
                raise KeyError(
                    f"experiment {self.name!r}: comparison reference "
                    f"{comparison.reference!r} is not an element of the {comparison.role!r} "
                    f"axis; have {list(axis.names())}"
                )

    def axes(self) -> dict[str, Axis]:
        return {
            ROLE_DATASET: self.datasets,
            ROLE_FEATURES: self.features,
            ROLE_MODEL: self.models,
            ROLE_POPULATION: self.populations,
            ROLE_SPLIT: self.splits,
        }

    def axis_for(self, role: str) -> Axis:
        try:
            return self.axes()[role]
        except KeyError:
            raise KeyError(f"unknown role {role!r}; known roles: {list(ROLES)}") from None

    def probe_budgets(self) -> tuple[float, ...]:
        return tuple(sorted({c.probe_budget for c in self.comparisons}))

    def cells(self) -> list[Cell]:
        """Every cell, ordered cheap-first by declared cost tier.

        The order is stable within a tier, so a cell's position is a function of the axis
        declaration rather than of anything measured.
        """
        axes = self.axes()
        rank = {tier: i for i, tier in enumerate(self.tier_order)}
        for axis in axes.values():
            for element in axis.elements:
                if element.tier not in rank:
                    raise ValueError(
                        f"experiment {self.name!r}: element {element.name!r} declares tier "
                        f"{element.tier!r}, which is not in tier_order {list(self.tier_order)}"
                    )
        cells: list[Cell] = []
        for combo in itertools.product(*(axes[role].elements for role in ROLES)):
            seconds = [e.estimated_seconds for e in combo if e.estimated_seconds is not None]
            cells.append(
                Cell(
                    factors=tuple((role, e.name) for role, e in zip(ROLES, combo)),
                    tier=max((e.tier for e in combo), key=lambda t: rank[t]),
                    estimated_seconds=sum(seconds) if seconds else None,
                )
            )
        cells.sort(key=lambda c: rank[c.tier])
        return cells

    def declaration(self) -> dict:
        return {
            "name": self.name,
            "note": self.note,
            "knobs": self.knobs.to_dict(),
            "history": self.history,
            "tier_order": list(self.tier_order),
            "axes": {role: axis.declaration() for role, axis in self.axes().items()},
            "comparisons": [
                {
                    "role": c.role,
                    "reference": c.reference,
                    "probe_budget": c.probe_budget,
                    "note": c.note,
                }
                for c in self.comparisons
            ],
        }


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
    spellings are ``artifact:<path>`` and a :class:`~rts.contract.Requirement` value, and
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


def _model_requirement(selector: models.Selector, ds: Dataset, env: Environment) -> Unmeasured | None:
    requirements = getattr(selector, "requirements", None)
    if requirements is None:
        return None
    for requirement in requirements():
        reason = unresolved(requirement, ds, env.caches)
        if reason is not None:
            return reason
    return None


# --- running ---------------------------------------------------------------


@dataclass(frozen=True)
class CellResult:
    """A measured cell: the numbers, and everything needed to read them."""

    cell: Cell
    selector: str
    population: str
    n_rows: int
    n_changes: int
    dataset_declaration: dict
    split: dict
    features: dict
    warnings: tuple[dict, ...]
    audit: tuple[dict, ...]
    results: list[dict]
    seconds: float = 0.0
    importances: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            **self.cell.to_dict(),
            "measured": True,
            "selector": self.selector,
            "population": self.population,
            "n_rows": self.n_rows,
            "n_changes": self.n_changes,
            "dataset_declaration": self.dataset_declaration,
            "split": self.split,
            "features": self.features,
            "warnings": list(self.warnings),
            "audit": list(self.audit),
            "results": self.results,
            "seconds": self.seconds,
            "importances": self.importances,
        }


@dataclass
class RunReport:
    """What a run produced. The experienced object, as opposed to the declaration."""

    experiment: str
    note: str
    environment: Environment
    axes: dict[str, list[dict]]
    comparisons_declared: list[dict]
    cells: list[CellResult] = field(default_factory=list)
    unmeasured: list[dict] = field(default_factory=list)
    comparisons: list[dict] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def n_cells(self) -> int:
        return len(self.cells) + len(self.unmeasured)

    def measured_cells(self) -> list[CellResult]:
        return list(self.cells)

    def find(self, **factors: str) -> CellResult:
        """The measured cell with these factor names. For tests and ad-hoc queries."""
        wanted = {r: n for r, n in factors.items()}
        for result in self.cells:
            have = result.cell.factors_dict()
            if all(have.get(role) == name for role, name in wanted.items()):
                return result
        raise KeyError(f"no measured cell matches {wanted!r}")

    def to_dict(self) -> dict:
        return {
            "experiment": self.experiment,
            "note": self.note,
            "environment": self.environment.to_dict(),
            "axes": self.axes,
            "comparisons_declared": self.comparisons_declared,
            "n_cells": self.n_cells,
            "cells": [c.to_dict() for c in self.cells],
            "unmeasured": self.unmeasured,
            "comparisons": self.comparisons,
            "seconds": self.seconds,
        }

    def save(self, out_dir: Path | str | None = None, stem: str | None = None) -> Path:
        target = Path(out_dir) if out_dir is not None else self.environment.out_dir
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"{stem or self.experiment}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    def format_table(self) -> str:
        """A compact per-cell table of recall at each budget."""
        if not self.cells:
            return "(no measured cells)"
        budgets = [r["budget"] for r in self.cells[0].results]
        width = max(len(c.cell.key) for c in self.cells)
        head = "  ".join(f"b{b:.2f}" for b in budgets)
        lines = [f"  {'cell'.ljust(width)}  {head}"]
        for result in self.cells:
            cells = "  ".join(f"{r['recall']:.3f}" for r in result.results)
            lines.append(f"  {result.cell.key.ljust(width)}  {cells}")
        return "\n".join(lines)


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
    selector: models.Selector,
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

    ctx = models.Context(
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
