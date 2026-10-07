"""What an experiment *is*, as a value: roles, factors, levels, design points, controls.

Nothing here measures anything. ``Experiment`` is an immutable declaration, ``Factor`` is an
ordered set of variants of one role, and ``Level`` is one variant with its cost -- and, when
a variant cannot be expressed for a role's value alone, its applicability. Measuring is
:mod:`.run`; what a measurement records is :mod:`.report`. The package docstring is the
original module's, which explains the five roles and why a design point is not a product.
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


#: The roles an experiment sweeps, in the order a design point's key names them.
FACTORS: tuple[str, ...] = (FACTOR_DATASET, FACTOR_FEATURES, FACTOR_MODEL, FACTOR_SUBSET, FACTOR_SPLIT)


#: Roles whose variation changes *which rows exist* or which pairs are rankable. Two design points
#: differing in one of these are not two measurements of one quantity, so a paired
#: contrast across them is refused rather than reported (``docs/experiment.md`` §7).
ROW_CHANGING_FACTORS = frozenset({FACTOR_DATASET, FACTOR_SUBSET, FACTOR_SPLIT})


ARTIFACT_PREFIX = "artifact:"



# --- controls and environment -------------------------------------------------


@dataclass(frozen=True)
class Controls:
    """Run-level constants: a choice that is free, not one the harness can derive.

    These do not multiply into design points. They parameterise every design point and are recorded once, so
    a difference in a result can never be attributed to a control that moved silently.
    """

    seed: int = config.SEED
    budgets: tuple[float, ...] = config.DEFAULT_BUDGETS
    n_bootstrap: int = config.DEFAULT_BOOTSTRAP
    candidate_policy: str = "full"
    #: Resamples for a paired test, when it differs from the table CI's. Two quantities, two
    #: precision needs: the recorded ladder used 1000 for its table intervals and 2000 for its
    #: paired p-values. ``None`` means "the same as ``n_bootstrap``".
    n_bootstrap_paired: int | None = None
    #: Seed for the *model's* randomness, when that should differ from the run's. ``seed`` fixes
    #: the synthetic change order and the temporal split, and a score cache is keyed to that order
    #: -- so a study that wants to re-fit one model under several seeds must vary this one and
    #: hold ``seed``, or it would move the data underneath the cache and invalidate the paired
    #: contrast instead of testing it. ``None`` means "the same as ``seed``".
    model_seed: int | None = None

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "model_seed": self.model_seed,
            "budgets": list(self.budgets),
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
    """What an :class:`Level` is handed when it is built.

    ``dataset`` is the design point's resolved dataset for every role except ``dataset`` itself, which
    is built first and sees ``None``. A dataset level that derives from another dataset
    simply constructs or shares its base inside ``make`` -- no extra machinery is needed,
    because ``make`` is already arbitrary Python.

    ``factors`` is the design point's factor names by role, and is populated **only when a level's
    applicability is checked**. It is empty while a value is being built, because a value is
    built once per run and shared across every design point that selects it: a level whose ``make``
    read this would silently be reused for design points it does not describe. Anything that genuinely
    varies by design point belongs in ``Level.applies``, which is asked per design point.
    """

    env: Environment
    dataset: Dataset | None = None
    factors: Mapping[str, str] = field(default_factory=dict)

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

    ``applies`` is for the case where availability cannot be asked of one role's value alone.
    Normally a requirement is a property of a value (``Ranker.requirements()``,
    ``Subset.needs``), and the kernel resolves it against the dataset. But some variants
    are only meaningful in combination with a particular level of *another* role -- a score
    artifact that covers one subset and not another -- and that is a fact the study config
    knows and no single value does. Declaring it here keeps the knowledge beside the variant
    rather than putting a special case in the kernel.
    """

    name: str
    make: Callable[[Binding], Any]
    tier: str = "cpu"
    estimated_seconds: float | None = None
    note: str = ""
    applies: Callable[[Binding], Undefined | None] | None = None

    def build(self, binding: Binding) -> Any:
        """The value this level denotes, or an :class:`Undefined` if it cannot exist."""
        return self.make(binding)

    def check(self, binding: Binding) -> Undefined | None:
        """Why this level does not apply to ``binding``'s design point, or ``None`` if it does."""
        if self.applies is None:
            return None
        return self.applies(binding)

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "tier": self.tier,
            "estimated_seconds": self.estimated_seconds,
            "note": self.note,
            "applies": self.applies is not None,
        }



def constant(name: str, obj: Any, **kwargs: Any) -> Level:
    """A level whose value does not depend on the run.

    The value is shared by every design point that selects this level, which is intended: for a
    dataset it means the artifact is read once however many models are swept over it.
    """
    return Level(name=name, make=lambda _binding: obj, **kwargs)



def unavailable(name: str, reason: Undefined, **kwargs: Any) -> Level:
    """A level that can never be materialised, carrying why.

    Used by :meth:`Factor.map` to keep a level whose shared option cannot be expressed for
    it: the design point is then reported undefined rather than silently absent.
    """
    return Level(
        name=name,
        make=lambda _binding, reason=reason: reason,
        note=kwargs.pop("note", "") or reason.note,
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

    ``history`` is an **override**, not an option. Whether the cumulative temporal columns are
    present is derived from the dataset's ``ordering()`` and from whether the split shuffles
    (``splits.Split.effective_order``); the harness warns when a caller overrides it. It is
    exposed because the study's recorded condition does override it deliberately, and the diagnostic
    travels with the design point.
    """

    name: str
    datasets: Factor
    features: Factor
    models: Factor
    subsets: Factor
    splits: Factor
    controls: Controls = Controls()
    contrasts: tuple[Contrast, ...] = ()
    temporal: bool | None = None
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
        for contrast in self.contrasts:
            factor = self.factors()[contrast.role]
            if not factor.has(contrast.reference):
                raise KeyError(
                    f"experiment {self.name!r}: contrast reference "
                    f"{contrast.reference!r} is not a level of the {contrast.role!r} "
                    f"factor; have {list(factor.names())}"
                )

    def factors(self) -> dict[str, Factor]:
        return {
            FACTOR_DATASET: self.datasets,
            FACTOR_FEATURES: self.features,
            FACTOR_MODEL: self.models,
            FACTOR_SUBSET: self.subsets,
            FACTOR_SPLIT: self.splits,
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
            "temporal": self.temporal,
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
