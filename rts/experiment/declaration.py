"""What an experiment *is*, as a value: roles, axes, elements, cells, knobs.

Nothing here measures anything. ``Experiment`` is an immutable declaration, ``Axis`` is an
ordered set of variants of one role, and ``Element`` is one variant with its cost -- and, when
a variant cannot be expressed for a role's value alone, its applicability. Measuring is
:mod:`.run`; what a measurement records is :mod:`.report`. The package docstring is the
original module's, which explains the five roles and why a cell is not a product.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import config
from ..data.contract import Dataset, Unmeasured, is_unmeasured

ROLE_DATASET = "dataset"

ROLE_FEATURES = "features"

ROLE_MODEL = "model"

ROLE_POPULATION = "population"

ROLE_SPLIT = "split"


#: The roles an experiment sweeps, in the order a cell's key names them.
ROLES: tuple[str, ...] = (ROLE_DATASET, ROLE_FEATURES, ROLE_MODEL, ROLE_POPULATION, ROLE_SPLIT)


#: Roles whose variation changes *which rows exist* or which pairs are rankable. Two cells
#: differing in one of these are not two measurements of one quantity, so a paired
#: comparison across them is refused rather than reported (``docs/experiment.md`` §7).
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

    def select(self, *names: str) -> Axis:
        """This axis restricted to ``names``, in the axis's own order."""
        wanted = set(names)
        unknown = wanted - set(self.names())
        if unknown:
            raise KeyError(f"axis {self.role!r} has no element(s) {sorted(unknown)}")
        return Axis(self.role, tuple(e for e in self.elements if e.name in wanted), self.note)

    def without(self, *names: str) -> Axis:
        """This axis with ``names`` removed."""
        dropped = set(names)
        unknown = dropped - set(self.names())
        if unknown:
            raise KeyError(f"axis {self.role!r} has no element(s) {sorted(unknown)}")
        return Axis(self.role, tuple(e for e in self.elements if e.name not in dropped), self.note)

    def extend(self, *elements: Element) -> Axis:
        return Axis(self.role, self.elements + tuple(elements), self.note)

    def map(self, fn: Callable[[Element], Element | Unmeasured]) -> Axis:
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
