"""What an experiment *is*, as a value: roles, factors, levels, design points, controls.

Nothing here measures anything. ``Experiment`` is an immutable declaration, ``Factor`` is an
ordered set of variants of one role, and ``Level`` is one variant with its cost. Measuring is
:mod:`.run`; what a measurement records is :mod:`.report`.

**A declaration is data, not code.** A level does not carry a closure; it carries a
:class:`Builder` -- a reusable factory, defined once beside the value it builds, plus the
arguments to call it with. So a study's declaration names pieces and supplies parameters, and
the only logic is in the pieces. ``constant`` and ``split_level`` are the kernel's own builders.

**Six roles.** An experiment sweeps ``dataset``, ``features``, ``model``, ``subset``, ``split``
and ``budget``. A new *level* in any role is cheap -- that is what most new experimental
dimensions are. A new *role* is a deliberate kernel change, because a role has to say what it
feeds. ``budget`` earns its place because it is genuinely swept: a metric is reported at each
budget of each change's own candidate set, and two studies that report different budget sets are
then two levels of one role rather than two runs that cannot be compared.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import config
from ..data.contract import Dataset, Undefined, is_undefined

FACTOR_DATASET = "dataset"

FACTOR_FEATURES = "features"

FACTOR_MODEL = "model"

FACTOR_SUBSET = "subset"

FACTOR_SPLIT = "split"

FACTOR_BUDGET = "budget"


#: The roles an experiment sweeps, in the order a design point's key names them.
FACTORS: tuple[str, ...] = (
    FACTOR_DATASET,
    FACTOR_FEATURES,
    FACTOR_MODEL,
    FACTOR_SUBSET,
    FACTOR_SPLIT,
    FACTOR_BUDGET,
)


#: Roles whose variation changes *which rows exist* or which pairs are rankable. Two design points
#: differing in one of these are not two measurements of one quantity, so a paired
#: contrast across them is refused rather than reported.
ROW_CHANGING_FACTORS = frozenset({FACTOR_DATASET, FACTOR_SUBSET, FACTOR_SPLIT})


ARTIFACT_PREFIX = "artifact:"


def _qualified(fn: Callable[..., Any]) -> str:
    """A readable name for a factory, for a report to record what a level was built from."""
    module = getattr(fn, "__module__", "")
    qualname = getattr(fn, "__qualname__", repr(fn))
    return f"{module}.{qualname}" if module else qualname


def _as_given(_binding: Binding, value: Any) -> Any:
    """A factory that returns its argument: the builder behind every constant level."""
    return value


def _jsonable(value: Any) -> Any:
    """A value a report can serialise: a primitive as-is, anything else by its repr.

    A builder's parameters are recorded so a report can say what a level was built from. A
    parameter may be an arbitrary object -- a dataset instance, say -- which is not JSON, so it
    is recorded as its repr rather than dropped.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


# --- builders ---------------------------------------------------------------


@dataclass(frozen=True)
class Builder:
    """How a level's value is made: a reusable factory plus the arguments to call it with.

    A *declaration* must contain no logic, so a level does not carry a closure written in the
    declaration. It carries a factory -- a function defined once, beside the value it builds --
    and a mapping of arguments. Because the arguments are data, a report can state what a level
    was built from, and two levels that differ only in a parameter are visibly different.
    """

    factory: Callable[..., Any]
    params: Mapping[str, Any] = field(default_factory=dict)

    def __call__(self, binding: Binding) -> Any:
        return self.factory(binding, **self.params)

    def metadata(self) -> dict:
        return {
            "factory": _qualified(self.factory),
            "params": {k: _jsonable(v) for k, v in self.params.items()},
        }


# --- controls and environment -------------------------------------------------


@dataclass(frozen=True)
class Controls:
    """Run-level constants: a choice that is free, not one the harness can derive.

    These do not multiply into design points. They parameterise every design point and are
    recorded once, so a difference in a result can never be attributed to a control that moved
    silently. Anything genuinely *swept* is a role, not a control -- which is why the budget is a
    factor rather than a member of this dataclass.
    """

    seed: int = config.SEED
    n_bootstrap: int = config.DEFAULT_BOOTSTRAP
    candidate_policy: str = "full"
    #: Resamples for a paired test, when it differs from the table CI's. Two quantities, two
    #: precision needs: a table interval and a paired p-value may want different counts.
    #: ``None`` means "the same as ``n_bootstrap``".
    n_bootstrap_paired: int | None = None
    #: Seed for the *model's* randomness, when that should differ from the run's. ``seed`` fixes
    #: the change order and the split, and a score cache is keyed to that order -- so a study that
    #: wants to re-fit one model under several seeds must vary this one and hold ``seed``, or it
    #: would move the data underneath the cache and invalidate the paired contrast instead of
    #: testing it. ``None`` means "the same as ``seed``".
    model_seed: int | None = None

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "model_seed": self.model_seed,
            "bootstrap_resamples": self.n_bootstrap,
            "paired_resample_count": self.paired_resamples,
            "candidate_policy": self.candidate_policy,
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
    """What a run offers to the levels it builds.

    ``caches`` maps a short name to an artifact path, so a run can point at a different cache
    without editing the experiment that names the requirement. ``shared`` is free-form and
    exists so a *caller* can inject an object a level expects -- a stub dataset in a test,
    or one expensive base dataset shared by several derived levels.
    """

    controls: Controls = Controls()
    out_dir: Path = config.ARTIFACTS
    caches: Mapping[str, Path] = field(default_factory=dict)
    shared: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "controls": self.controls.to_dict(),
            "out_dir": str(self.out_dir),
            "caches": {k: str(v) for k, v in self.caches.items()},
            "shared": sorted(self.shared),
        }


@dataclass(frozen=True)
class Binding:
    """What a level is handed when it is built.

    ``dataset`` is the design point's resolved dataset for every role except ``dataset`` itself,
    which is built first and sees ``None``. A dataset level that derives from another dataset
    constructs or shares its base through its builder's parameters -- the parameters are data, so
    the derivation is visible rather than hidden in a closure.

    Nothing here varies *per design point*: a value is built once per run and shared across every
    design point that selects it. Availability that depends on another role is a property of the
    value (its ``requirements()``), resolved by the run against the dataset -- not a per-design-point
    predicate.
    """

    env: Environment
    dataset: Dataset | None = None

    @property
    def controls(self) -> Controls:
        return self.env.controls

    @property
    def caches(self) -> Mapping[str, Path]:
        return self.env.caches

    @property
    def shared(self) -> Mapping[str, Any]:
        return self.env.shared


# --- levels and factors -----------------------------------------------------


@dataclass(frozen=True)
class Level:
    """One variant of one input: a name, a builder, and what it costs to measure.

    ``builder`` is a :class:`Builder` -- a factory and its parameters -- rather than an arbitrary
    callable, so a level's value is described by data. Use :meth:`Level.of` to declare one, or
    :func:`constant` for a value that does not depend on the run.
    """

    name: str
    builder: Builder
    tier: str = "cpu"
    estimated_seconds: float | None = None
    note: str = ""

    def build(self, binding: Binding) -> Any:
        """The value this level denotes, or an :class:`Undefined` if it cannot exist."""
        return self.builder(binding)

    @classmethod
    def of(
        cls,
        factory: Callable[..., Any],
        name: str,
        *,
        tier: str = "cpu",
        estimated_seconds: float | None = None,
        note: str = "",
        **params: Any,
    ) -> Level:
        """A level built by ``factory(binding, **params)``.

        ``factory`` is a reusable constructor -- a module-level function, or a classmethod -- so a
        declaration that calls this names a piece rather than writing logic.
        """
        return cls(
            name=name,
            builder=Builder(factory, params),
            tier=tier,
            estimated_seconds=estimated_seconds,
            note=note,
        )

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "tier": self.tier,
            "estimated_seconds": self.estimated_seconds,
            "note": self.note,
            "builder": self.builder.metadata(),
        }


def constant(name: str, obj: Any, **kwargs: Any) -> Level:
    """A level whose value does not depend on the run.

    The value is shared by every design point that selects this level, which is intended: for a
    dataset it means the artifact is read once however many models are swept over it.
    """
    return Level(name=name, builder=Builder(_as_given, {"value": obj}), **kwargs)


def unavailable(name: str, reason: Undefined, **kwargs: Any) -> Level:
    """A level that can never be materialised, carrying why.

    Used by :meth:`Factor.map` to keep a level whose shared option cannot be expressed for
    it: the design point is then reported undefined rather than silently absent.
    """
    return Level(
        name=name,
        builder=Builder(_as_given, {"value": reason}),
        note=kwargs.pop("note", "") or reason.note,
        **kwargs,
    )


def _make_split(binding: Binding, train_fraction: float, shuffle: bool):
    from ..data import splits

    return splits.make_split(
        binding.dataset,
        train_fraction=train_fraction,
        shuffle=shuffle,
        seed=binding.controls.seed,
    )


def split_level(
    name: str | None = None,
    *,
    train_fraction: float = config.DEFAULT_TRAIN_FRACTION,
    shuffle: bool = False,
    **kwargs: Any,
) -> Level:
    """A declared train/test split, as a reusable constructor rather than a lambda.

    The split is evaluation configuration the experiment chooses, and it is the one role whose
    value depends on another role (the dataset). This constructor is what keeps a *declaration*
    free of logic: a study writes ``split_level(train_fraction=0.5)`` and nothing else.
    """
    label = name or f"split{int(round(train_fraction * 100))}" + (
        "_shuffled" if shuffle else ""
    )
    return Level(
        label,
        Builder(_make_split, {"train_fraction": train_fraction, "shuffle": shuffle}),
        note=f"{train_fraction:.0%} train prefix" + (", shuffled" if shuffle else ""),
        **kwargs,
    )


def budget_level(budget: float, name: str | None = None, **kwargs: Any) -> Level:
    """One evaluation budget, as a swept level.

    A budget is the fraction of *each change's own* candidate set that a metric may select, so
    two budgets are two quantities rather than one quantity at two thresholds. Making it a level
    is what lets a study sweep several at once and report each on its own.
    """
    return Level(
        name or f"b{budget:g}",
        Builder(_as_given, {"value": budget}),
        note=f"select ceil({budget:g} x the change's own candidate count)",
        **kwargs,
    )


@dataclass(frozen=True)
class Factor:
    """A named, ordered set of levels for one role."""

    role: str
    levels: tuple[Level, ...]
    note: str = ""

    def __post_init__(self) -> None:
        if self.role not in FACTORS:
            raise ValueError(f"unknown role {self.role!r}; known roles: {list(FACTORS)}")
        names = [e.name for e in self.levels]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"factor {self.role!r} has duplicate level names: {duplicates}")
        if not self.levels:
            raise ValueError(f"factor {self.role!r} has no levels")

    def names(self) -> tuple[str, ...]:
        return tuple(e.name for e in self.levels)

    def get(self, name: str) -> Level:
        for level in self.levels:
            if level.name == name:
                return level
        raise KeyError(f"factor {self.role!r} has no level {name!r}; have {list(self.names())}")

    def has(self, name: str) -> bool:
        return any(e.name == name for e in self.levels)

    def select(self, *names: str) -> Factor:
        """This factor restricted to ``names``, in the factor's own order."""
        wanted = set(names)
        unknown = wanted - set(self.names())
        if unknown:
            raise KeyError(f"factor {self.role!r} has no level(s) {sorted(unknown)}")
        return Factor(self.role, tuple(e for e in self.levels if e.name in wanted), self.note)

    def without(self, *names: str) -> Factor:
        """This factor with ``names`` removed."""
        dropped = set(names)
        unknown = dropped - set(self.names())
        if unknown:
            raise KeyError(f"factor {self.role!r} has no level(s) {sorted(unknown)}")
        return Factor(self.role, tuple(e for e in self.levels if e.name not in dropped), self.note)

    def extend(self, *levels: Level) -> Factor:
        return Factor(self.role, self.levels + tuple(levels), self.note)

    def map(self, fn: Callable[[Level], Level | Undefined]) -> Factor:
        """Apply a shared dimension-level option to every level.

        ``fn`` returns the level as it should be with the option applied, or an
        :class:`Undefined` when the option cannot be expressed for that level -- which is a
        real case, not a degenerate one: "exclude the temporal family" is expressible for a
        column-reading model and meaningless for one that reads text pairs. An inapplicable
        level is kept as an undefined level rather than dropped, so its design points appear as
        undefined findings.
        """
        out: list[Level] = []
        for level in self.levels:
            result = fn(level)
            if is_undefined(result):
                out.append(
                    unavailable(level.name, result, tier=level.tier, note=level.note)
                )
            else:
                out.append(result)
        return Factor(self.role, tuple(out), self.note)

    def metadata(self) -> list[dict]:
        return [e.metadata() for e in self.levels]


def declared_budgets(factor: Factor) -> tuple[float, ...]:
    """The budget values a budget factor declares.

    A budget level is a constant (see :func:`budget_level`), so the values are read straight off
    its builder's parameters rather than by building it.
    """
    return tuple(level.builder.params["value"] for level in factor.levels)


# --- design points and contrasts -------------------------------------------------


@dataclass(frozen=True)
class DesignPoint:
    """One point in the product: one level per role, plus the run's controls."""

    factors: tuple[tuple[str, str], ...]
    tier: str = "cpu"
    estimated_seconds: float | None = None

    def name(self, role: str) -> str:
        for r, n in self.factors:
            if r == role:
                return n
        raise KeyError(f"design_point has no factor for role {role!r}")

    @property
    def key(self) -> str:
        """Stable identity of the design point: the level names, in role order."""
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
class Contrast:
    """A paired delta against a reference level of one role, at a probe budget.

    Only a *pairable* role may be named. Pairing is a claim that the two design points measure the same
    quantity on the same rows, which is false the moment the varying role changes the rows --
    so a row-changing role is refused at construction rather than producing a plausible number
    from two different subsets.
    """

    role: str
    reference: str
    probe_budget: float
    note: str = ""

    def __post_init__(self) -> None:
        if self.role in ROW_CHANGING_FACTORS:
            pairable = sorted(set(FACTORS) - ROW_CHANGING_FACTORS)
            raise ValueError(
                f"contrast over role {self.role!r} would pair different subsets: "
                f"that role changes which rows exist, so only {pairable} give a paired delta"
            )


# --- the experiment --------------------------------------------------------


@dataclass(frozen=True)
class Experiment:
    """A declaration of what to sweep. A value: building one runs nothing.

    Whether the cumulative temporal columns are present is **derived**, not configured: it
    follows from the dataset's ``ordering()`` and from whether the split shuffles
    (``splits.Split.effective_order``). There is deliberately no override -- a switch here would
    let a run present a feature set the data does not support.
    """

    name: str
    datasets: Factor
    features: Factor
    models: Factor
    subsets: Factor
    splits: Factor
    budgets: Factor
    controls: Controls = Controls()
    contrasts: tuple[Contrast, ...] = ()
    tier_order: tuple[str, ...] = ("cpu", "gpu")
    note: str = ""

    def __post_init__(self) -> None:
        for role, factor in self.factors().items():
            if factor.role != role:
                raise ValueError(
                    f"experiment {self.name!r}: factor for {role!r} declares role {factor.role!r}"
                )
        if not self.tier_order:
            raise ValueError(f"experiment {self.name!r}: tier_order is empty")
        declared = set(declared_budgets(self.budgets))
        for contrast in self.contrasts:
            factor = self.factors()[contrast.role]
            if not factor.has(contrast.reference):
                raise KeyError(
                    f"experiment {self.name!r}: contrast reference "
                    f"{contrast.reference!r} is not a level of the {contrast.role!r} "
                    f"factor; have {list(factor.names())}"
                )
            if contrast.probe_budget not in declared:
                raise ValueError(
                    f"experiment {self.name!r}: contrast probe budget "
                    f"{contrast.probe_budget} is not a declared budget level; have "
                    f"{sorted(declared)}"
                )

    def factors(self) -> dict[str, Factor]:
        return {
            FACTOR_DATASET: self.datasets,
            FACTOR_FEATURES: self.features,
            FACTOR_MODEL: self.models,
            FACTOR_SUBSET: self.subsets,
            FACTOR_SPLIT: self.splits,
            FACTOR_BUDGET: self.budgets,
        }

    def factor_for(self, role: str) -> Factor:
        try:
            return self.factors()[role]
        except KeyError:
            raise KeyError(f"unknown role {role!r}; known roles: {list(FACTORS)}") from None

    def probe_budgets(self) -> tuple[float, ...]:
        return tuple(sorted({c.probe_budget for c in self.contrasts}))

    def design_points(self) -> list[DesignPoint]:
        """Every design point, ordered cheap-first by declared cost tier.

        The order is stable within a tier, so a design point's position is a function of the
        factor declaration rather than of anything measured.
        """
        factors = self.factors()
        rank = {tier: i for i, tier in enumerate(self.tier_order)}
        for factor in factors.values():
            for level in factor.levels:
                if level.tier not in rank:
                    raise ValueError(
                        f"experiment {self.name!r}: level {level.name!r} declares tier "
                        f"{level.tier!r}, which is not in tier_order {list(self.tier_order)}"
                    )
        design_points: list[DesignPoint] = []
        for combo in itertools.product(*(factors[role].levels for role in FACTORS)):
            seconds = [e.estimated_seconds for e in combo if e.estimated_seconds is not None]
            design_points.append(
                DesignPoint(
                    factors=tuple((role, e.name) for role, e in zip(FACTORS, combo)),
                    tier=max((e.tier for e in combo), key=lambda t: rank[t]),
                    estimated_seconds=sum(seconds) if seconds else None,
                )
            )
        design_points.sort(key=lambda c: rank[c.tier])
        return design_points

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "note": self.note,
            "controls": self.controls.to_dict(),
            "tier_order": list(self.tier_order),
            "factors": {role: factor.metadata() for role, factor in self.factors().items()},
            "contrasts": [
                {
                    "role": c.role,
                    "reference": c.reference,
                    "probe_budget": c.probe_budget,
                    "note": c.note,
                }
                for c in self.contrasts
            ],
        }
