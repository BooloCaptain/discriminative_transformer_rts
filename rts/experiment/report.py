"""What a run records: one measured design point, and the report over all of them.

``DesignPointResult`` carries the numbers *and* everything needed to read them -- the design point's split,
the dataset's metadata, the feature audit, the two diagnostic vocabularies kept apart --
because a renderer that has to rebuild any of that can describe a dataset the run did not
measure. ``RunReport`` is the experienced object, and it reads its own records back
(``describe``, ``declaration``, ``recurrence``, ``subset_size``), refusing to guess which
dataset was meant when a run swept more than one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .declaration import DesignPoint, Environment

# --- running ---------------------------------------------------------------


@dataclass(frozen=True)
class DesignPointResult:
    """A measured design point: the numbers, and everything needed to read them."""

    design_point: DesignPoint
    ranker: str
    subset: str
    n_rows: int
    n_changes: int
    #: The subset's rows **in the evaluation window**, before the fault filter.
    #: ``n_rows`` is what the metric actually averaged over (the fault-bearing ones); both
    #: are recorded because a renderer that reported one as the other would misstate how
    #: much data a condition rests on.
    n_population_rows: int
    dataset_metadata: dict
    split: dict
    features: dict
    diagnostics: tuple[dict, ...]
    audit: tuple[dict, ...]
    results: list[dict]
    seconds: float = 0.0
    importances: dict = field(default_factory=dict)
    #: What a producing ranker spent, if it produced (pairs scored, throughput, read-vs-produced).
    #: Empty for a ranker that computes in-process, whose cost is ``seconds``.
    production: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            **self.design_point.to_dict(),
            "measured": True,
            "ranker": self.ranker,
            "subset": self.subset,
            "n_rows": self.n_rows,
            "n_changes": self.n_changes,
            "n_population_rows": self.n_population_rows,
            "dataset_metadata": self.dataset_metadata,
            "split": self.split,
            "features": self.features,
            "diagnostics": list(self.diagnostics),
            "audit": list(self.audit),
            "results": self.results,
            "seconds": self.seconds,
            "importances": self.importances,
            "production": self.production,
        }



@dataclass
class RunReport:
    """What a run produced. The experienced object, as opposed to the declaration."""

    experiment: str
    note: str
    environment: Environment
    factors: dict[str, list[dict]]
    comparisons_declared: list[dict]
    design_points: list[DesignPointResult] = field(default_factory=list)
    undefined: list[dict] = field(default_factory=list)
    contrasts: list[dict] = field(default_factory=list)
    #: ``"{dataset level}|{split level}"`` -> ``declaration``, ``describe``, ``recurrence``.
    #: Facts about the data the run measured, recorded once per (dataset, split) so that a
    #: renderer does not have to rebuild the dataset to describe it. ``recurrence`` is
    #: ``None`` for a dataset that declares no coverage, because the statistic is not
    #: defined without it.
    dataset_stats: dict[str, dict] = field(default_factory=dict)
    #: ``"{dataset level}|{subset}"`` -> ``{"changes": rows in the window,
    #: "faults": rows averaged over}``. Keyed by the *pair* rather than by the subset
    #: name alone: a run over two datasets can have a subset of the same name and
    #: different sizes in each, and reporting the first dataset's counts for the second is
    #: the same error as :meth:`describe` guessing which dataset was meant. A single-dataset
    #: run is unaffected, because then there is exactly one key per subset name.
    subset_sizes: dict[str, dict] = field(default_factory=dict)
    seconds: float = 0.0

    # --- reading the report back ------------------------------------------

    def _one(self, available, what: str):
        """The single entry of a keyed record, or a KeyError naming what there is.

        Most runs have exactly one dataset and one split, and the caller knows which. A run
        with several has to name one -- guessing would silently describe the wrong dataset.
        """
        if len(available) == 1:
            return next(iter(available.values()))
        if not available:
            raise KeyError(f"this report records no {what}")
        raise KeyError(
            f"this report records {len(available)} {what}; name one of {sorted(available)}"
        )

    def _stats(self, dataset: str | None, split: str | None) -> dict:
        """One ``dataset/split`` record: named, or implied when the run had only one."""
        if (dataset is None) != (split is None):
            raise ValueError("name both the dataset and the split, or neither")
        if dataset is None:
            return self._one(self.dataset_stats, "dataset/split pairs")
        key = f"{dataset}|{split}"
        if key not in self.dataset_stats:
            raise KeyError(
                f"this report records no dataset/split pair {key!r}; it has "
                f"{sorted(self.dataset_stats)}"
            )
        return self.dataset_stats[key]

    def describe(self, dataset: str | None = None, split: str | None = None) -> dict:
        """The dataset's shape under the split the run used (:func:`reporting.describe`)."""
        return self._stats(dataset, split)["describe"]

    def metadata(self, dataset: str | None = None, split: str | None = None) -> dict:
        """The dataset's own declaration, which a run records once per dataset."""
        return self._stats(dataset, split)["metadata"]

    def recurrence(self, dataset: str | None = None, split: str | None = None):
        """How often ``(file, test)`` pairs recur, or ``None`` if coverage is not declared."""
        return self._stats(dataset, split)["recurrence"]

    def subset_size(
        self, subset: str, dataset: str | None = None
    ) -> tuple[int, int]:
        """``(rows in the evaluation window, rows the metric averaged over)``.

        The second is the fault-bearing subset, so it is what every recall in the report was
        averaged over. Both come from the design points rather than from re-deriving the subset,
        so a renderer cannot report a size the run did not use.

        ``dataset`` names the dataset level, and is only needed when the run had more than
        one -- the same rule :meth:`describe` follows. A report that recorded this subset
        for several datasets refuses to guess, because returning the first one's counts is
        indistinguishable from a correct answer.
        """
        if dataset is None:
            keys = sorted(
                key for key in self.subset_sizes if key.endswith(f"|{subset}")
            )
            if not keys:
                raise KeyError(
                    f"this report records no sizes for subset {subset!r}; it has "
                    f"{sorted(self.subset_sizes)}"
                )
            if len(keys) > 1:
                raise KeyError(
                    f"this report records subset {subset!r} for several datasets "
                    f"{[key.split('|', 1)[0] for key in keys]}; name which one"
                )
            entry = self.subset_sizes[keys[0]]
        else:
            key = f"{dataset}|{subset}"
            if key not in self.subset_sizes:
                raise KeyError(
                    f"this report records no subset {subset!r} for dataset "
                    f"{dataset!r}; it has {sorted(self.subset_sizes)}"
                )
            entry = self.subset_sizes[key]
        return entry["changes"], entry["faults"]

    @property
    def n_cells(self) -> int:
        return len(self.design_points) + len(self.undefined)

    def measured_cells(self) -> list[DesignPointResult]:
        return list(self.design_points)

    def find(self, **factors: str) -> DesignPointResult:
        """The measured design point with these factor names. For tests and ad-hoc queries."""
        wanted = {r: n for r, n in factors.items()}
        for result in self.design_points:
            have = result.design_point.factors_dict()
            if all(have.get(role) == name for role, name in wanted.items()):
                return result
        raise KeyError(f"no measured design_point matches {wanted!r}")

    def to_dict(self) -> dict:
        return {
            "experiment": self.experiment,
            "note": self.note,
            "environment": self.environment.to_dict(),
            "factors": self.factors,
            "comparisons_declared": self.comparisons_declared,
            "n_cells": self.n_cells,
            "design_points": [c.to_dict() for c in self.design_points],
            "undefined": self.undefined,
            "contrasts": self.contrasts,
            "dataset_stats": self.dataset_stats,
            "subset_sizes": self.subset_sizes,
            "seconds": self.seconds,
        }

    def save(self, out_dir: Path | str | None = None, stem: str | None = None) -> Path:
        target = Path(out_dir) if out_dir is not None else self.environment.out_dir
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"{stem or self.experiment}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    def format_table(self) -> str:
        """A compact table: one line per measured design point, with its budget and recall."""
        if not self.design_points:
            return "(no measured design_points)"
        width = max(len(c.design_point.key) for c in self.design_points)
        lines = [f"  {'design_point'.ljust(width)}  {'budget':>6}  recall"]
        for result in self.design_points:
            for row in result.results:
                lines.append(
                    f"  {result.design_point.key.ljust(width)}  {row['budget']:>6.2f}  "
                    f"{row['recall']:.3f}"
                )
        return "\n".join(lines)
