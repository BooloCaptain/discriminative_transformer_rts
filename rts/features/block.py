"""Feature blocks: an ordered declaration of columns, and the matrix they build.

The distinction this module exists to make:

* A **derived feature** is one quantity computed from the contract (``change_size``,
  ``path_distance``, ...). Individually addressable, individually testable.
* A **feature block** is an ordered, named *selection* of those quantities assembled
  into one ``[n_changes, n_tests, n_columns]`` tensor for a model input.

Colocating them is why the first pass could not add a feature without editing the
assembly function. Here, adding a column means adding it to a group in the block's own
module: one file, one line, no central registry.

The assembly returns a :class:`FeatureMatrix` rather than a bare ``(X, names)`` tuple,
which fixes the API gap that caused nine separate ``{n: i for i, n in enumerate(names)}``
reconstructions across the package. A consumer now asks the matrix for a column index
and gets a loud failure if the column was renamed, instead of silently reading the
neighbouring column.

Gating is derived, not asserted. A group names the *material* it reads
(``needs=("coverage",)``); :func:`rts.data.contract.requirements_for` maps that onto
requirements, and the block treats a group whose requirements the dataset cannot meet as
**unmeasured** -- zeroed at the model-input boundary, which is the one place a lossy
coercion is legitimate, with the reason recorded and returned. Nothing hand-writes a
requirement set that could drift from the code reading it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..data import accessors
from ..data.accessors import MATERIAL, requirements_for
from ..data.contract import Dataset, Requirement, Unmeasured, Warning, Warnings


@dataclass(frozen=True)
class FeatureColumn:
    """One column: a name, and the family the ablation ladder treats it as part of."""

    name: str
    family: str = "intrinsic"
    note: str = ""


def _project(
    produce: Callable[[Mapping[str, Any]], Sequence[np.ndarray]], positions: tuple[int, ...]
) -> Callable[[Mapping[str, Any]], tuple[np.ndarray, ...]]:
    """Wrap a group's computation to yield only the columns at ``positions``.

    Needed by :meth:`FeatureBlock.only`, which may keep part of a group: a group's
    computation produces all of its columns at once, so keeping one of three must project
    the tuple rather than re-derive it.
    """

    def projected(material: Mapping[str, Any]) -> tuple[np.ndarray, ...]:
        produced = tuple(produce(material))
        return tuple(produced[i] for i in positions)

    return projected


@dataclass(frozen=True)
class FeatureGroup:
    """One computation producing one or more columns.

    Grouping is by computation, not by family: the history block produces three columns
    from one pass and the coverage block three from one matrix, while the *traceability*
    family draws one column from each of two groups. Conflating the two would force a
    computation per column for no reason.

    A need naming material the contract does not know about must be supplied by the
    caller through ``build(..., extra=...)``. That is how a block over a *derived*
    dataset -- a bundle, which needs its members and its base's own features -- declares
    what it reads without adding bundle vocabulary to the contract's global list.
    """

    columns: tuple[FeatureColumn, ...]
    needs: tuple[str, ...]
    produce: Callable[[Mapping[str, Any]], Sequence[np.ndarray]]
    note: str = ""

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)

    def requires(self, external: frozenset[str] = frozenset()) -> frozenset[Requirement]:
        out: set[Requirement] = set()
        for name in self.needs:
            if name in MATERIAL:
                out |= MATERIAL[name].requires
            elif name not in external:
                raise KeyError(
                    f"group {self.names[0]!r} needs material {name!r}, which is neither "
                    f"contract material nor supplied by the caller; known material: "
                    f"{sorted(MATERIAL)}"
                )
        return frozenset(out)


@dataclass(frozen=True)
class FeatureMatrix:
    """A built feature tensor, with the coercion it required.

    ``unmeasured`` maps a column to why it could not be defined. Reporting it is the
    point: ``0.0`` from a coercion and ``0.0`` from a measurement are different claims,
    and only one of them is about the data.
    """

    X: np.ndarray
    columns: tuple[str, ...]
    unmeasured: tuple[tuple[str, Unmeasured], ...] = ()
    warnings: tuple[Warning, ...] = ()

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.X.shape  # type: ignore[return-value]

    def index(self, name: str) -> int:
        try:
            return self.columns.index(name)
        except ValueError:
            raise KeyError(
                f"no column {name!r} in this matrix; have {list(self.columns)}"
            ) from None

    def indices(self, names: Sequence[str]) -> tuple[int, ...]:
        return tuple(self.index(n) for n in names)

    def column(self, name: str) -> np.ndarray:
        return self.X[:, :, self.index(name)]

    def keep(self, names: Sequence[str]) -> np.ndarray:
        """``X`` restricted to ``names``, in the order given."""
        return self.X[:, :, list(self.indices(names))]

    def without(self, names: Sequence[str]) -> np.ndarray:
        """``X`` with ``names`` dropped, preserving the declared order."""
        dropped = set(names)
        return self.X[:, :, [i for i, c in enumerate(self.columns) if c not in dropped]]

    def measured(self, name: str) -> bool:
        return not self.is_unmeasured(name)

    def is_unmeasured(self, name: str) -> bool:
        return any(column == name for column, _ in self.unmeasured)

    def reason(self, name: str) -> Unmeasured | None:
        for column, reason in self.unmeasured:
            if column == name:
                return reason
        return None

    def audit(self) -> dict:
        return {
            "columns": list(self.columns),
            "unmeasured": [{"column": c, **r.to_dict()} for c, r in self.unmeasured],
        }


@dataclass(frozen=True)
class FeatureBlock:
    """An ordered set of feature groups, buildable against any dataset.

    ``suppressed`` names columns the caller withholds. The slots stay -- a rung of the
    ablation ladder must keep a stable column list, because selectors index columns by
    name -- but the columns are treated as unmeasured, so a suppression and a genuine
    absence take exactly the same path through the harness.
    """

    name: str
    groups: tuple[FeatureGroup, ...]
    suppressed: frozenset[str] = field(default_factory=frozenset)
    note: str = ""

    # --- declaration queries ---------------------------------------------

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(c.name for g in self.groups for c in g.columns)

    @property
    def families(self) -> tuple[str, ...]:
        seen: list[str] = []
        for group in self.groups:
            for column in group.columns:
                if column.family not in seen:
                    seen.append(column.family)
        return tuple(seen)

    def family(self, family: str) -> tuple[str, ...]:
        """The columns belonging to ``family``, in declared order."""
        return tuple(
            c.name for g in self.groups for c in g.columns if c.family == family
        )

    def has_family(self, family: str) -> bool:
        """Whether any column declares this family.

        Needed because withholding is strict -- a family name that matches nothing is an
        error, since a typo'd ablation would silently stop ablating anything -- so a
        caller that wants to withhold a family *if present* must ask first.
        """
        return any(c.family == family for g in self.groups for c in g.columns)

    def index(self, name: str) -> int:
        try:
            return self.columns.index(name)
        except ValueError:
            raise KeyError(f"no column {name!r} in block {self.name!r}") from None

    def indices(self, names: Sequence[str]) -> tuple[int, ...]:
        return tuple(self.index(n) for n in names)

    def requires(self, external: frozenset[str] = frozenset()) -> frozenset[Requirement]:
        """Every requirement any group in this block depends on."""
        out: set[Requirement] = set()
        for group in self.groups:
            out |= group.requires(external)
        return frozenset(out)

    @property
    def external_material(self) -> frozenset[str]:
        """Material names this block needs that the contract does not provide."""
        return frozenset(
            name
            for group in self.groups
            for name in group.needs
            if name not in MATERIAL
        )

    def group_of(self, name: str) -> FeatureGroup:
        for group in self.groups:
            if name in group.names:
                return group
        raise KeyError(f"no column {name!r} in block {self.name!r}")

    # --- derivation -------------------------------------------------------

    def without(self, *names: str) -> "FeatureBlock":
        """A block in which ``names`` are withheld. The slots are preserved."""
        unknown = [n for n in names if n not in self.columns]
        if unknown:
            raise KeyError(f"block {self.name!r} has no column(s) {unknown}")
        return FeatureBlock(
            name=self.name,
            groups=self.groups,
            suppressed=self.suppressed | frozenset(names),
            note=self.note,
        )

    def without_families(self, *families: str) -> "FeatureBlock":
        """A block in which every column of ``families`` is withheld."""
        names: list[str] = []
        for family in families:
            names.extend(self.family(family))
        if not names:
            raise KeyError(f"block {self.name!r} has no family among {families}")
        return self.without(*names)

    def only(self, names: Sequence[str]) -> "FeatureBlock":
        """A block holding only ``names``, in the declared order.

        A group may be *partially* kept, so its ``produce`` is wrapped to project the
        result down to the columns that survived: copying the group wholesale would make it
        return more columns than it declares, which ``build`` would reject.
        """
        keep = set(names)
        unknown = keep - set(self.columns)
        if unknown:
            raise KeyError(f"block {self.name!r} has no column(s) {sorted(unknown)}")
        groups: list[FeatureGroup] = []
        for group in self.groups:
            kept = tuple(c for c in group.columns if c.name in keep)
            if not kept:
                continue
            if len(kept) == len(group.columns):
                groups.append(group)
                continue
            positions = tuple(i for i, c in enumerate(group.columns) if c.name in keep)
            groups.append(
                FeatureGroup(
                    columns=kept,
                    needs=group.needs,
                    produce=_project(group.produce, positions),
                    note=group.note,
                )
            )
        return FeatureBlock(
            name=self.name, groups=tuple(groups), suppressed=self.suppressed, note=self.note
        )

    # --- build ------------------------------------------------------------

    def build(
        self,
        ds: Dataset,
        warnings: Warnings | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> FeatureMatrix:
        """Build ``[n_changes, n_tests, n_columns]`` against ``ds``.

        A group the dataset cannot satisfy, and a column the caller has withheld, both
        produce an all-zero block and an :class:`Unmeasured` entry naming what was
        missing. Silent substitution would be the alternative, and a model will happily
        split on a column that means "we could not measure this".

        ``extra`` supplies material the contract does not know about. A block that needs
        it and is not given it is a programming error, not an unmeasured quantity, so it
        raises rather than zeroing: the caller promised the material.
        """
        collected = warnings if warnings is not None else Warnings()
        supplied = dict(extra or {})
        absent = self.external_material - frozenset(supplied)
        if absent:
            raise ValueError(
                f"block {self.name!r} needs caller-supplied material {sorted(absent)}, "
                f"which build() was not given"
            )
        available = {**accessors.material(ds), **supplied}
        n_c, n_t = ds.n_changes, ds.n_tests
        n_f = len(self.columns)

        X = np.zeros((n_c, n_t, n_f), dtype=np.float32)
        unmeasured: list[tuple[str, Unmeasured]] = []
        cursor = 0

        for group in self.groups:
            missing = ds.missing_requirements(group.requires(self.external_material))
            if missing:
                reason = Unmeasured(
                    requirement=missing[0].value,
                    note=(
                        f"block {self.name!r} group {group.names[0]!r} needs "
                        f"{', '.join(m.value for m in missing)}, which dataset "
                        f"{ds.name!r} does not provide"
                    ),
                )
                for column in group.columns:
                    unmeasured.append((column.name, reason))
                collected.add(
                    "feature.unmeasured",
                    reason.requirement,
                    f"{', '.join(group.names)} unmeasured: {reason.note}; coerced to 0 at "
                    "the model-input boundary",
                    scope=f"block:{self.name}:{group.names[0]}",
                )
                cursor += len(group.columns)
                continue

            produced = list(group.produce(available))
            if len(produced) != len(group.columns):
                raise ValueError(
                    f"block {self.name!r} group {group.names[0]!r} declared "
                    f"{len(group.columns)} column(s) but produced {len(produced)}"
                )
            for column, values in zip(group.columns, produced):
                block = np.asarray(values, dtype=np.float32)
                if block.shape != (n_c, n_t):
                    # Per-test constants naturally come back as [1, n_t] and per-change ones
                    # as [n_c, 1]; accepting anything broadcastable keeps the group
                    # definitions honest about what they are, while still rejecting a shape
                    # that would silently misalign.
                    try:
                        block = np.broadcast_to(block, (n_c, n_t))
                    except ValueError:
                        raise ValueError(
                            f"block {self.name!r} column {column.name!r} produced shape "
                            f"{np.asarray(values).shape}, which does not broadcast to "
                            f"{(n_c, n_t)}"
                        ) from None
                if column.name in self.suppressed:
                    reason = Unmeasured(
                        requirement="withheld",
                        note=(
                            f"column {column.name!r} was withheld by the caller (an "
                            "ablation rung), so it is unmeasured rather than measured"
                        ),
                    )
                    unmeasured.append((column.name, reason))
                    collected.add(
                        "feature.withheld",
                        "withheld",
                        f"{column.name!r} withheld by the caller; coerced to 0 at the "
                        "model-input boundary",
                        scope=f"block:{self.name}:{column.name}",
                    )
                else:
                    X[:, :, cursor] = block
                cursor += 1

        return FeatureMatrix(
            X=X,
            columns=self.columns,
            unmeasured=tuple(unmeasured),
            warnings=tuple(collected),
        )


__all__ = ["FeatureBlock", "FeatureColumn", "FeatureGroup", "FeatureMatrix"]
