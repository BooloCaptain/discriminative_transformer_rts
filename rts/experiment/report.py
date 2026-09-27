"""What a run records: one measured cell, and the report over all of them.

``CellResult`` carries the numbers *and* everything needed to read them -- the cell's split,
the dataset's declaration, the feature audit, the two warning vocabularies kept apart --
because a renderer that has to rebuild any of that can describe a dataset the run did not
measure. ``RunReport`` is the experienced object, and it reads its own records back
(``describe``, ``declaration``, ``recurrence``, ``population_size``), refusing to guess which
dataset was meant when a run swept more than one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .declaration import Cell, Environment

# --- running ---------------------------------------------------------------


@dataclass(frozen=True)
class CellResult:
    """A measured cell: the numbers, and everything needed to read them."""

    cell: Cell
    selector: str
    population: str
    n_rows: int
    n_changes: int
    #: The population's rows **in the evaluation window**, before the fault filter.
    #: ``n_rows`` is what the metric actually averaged over (the fault-bearing ones); both
    #: are recorded because a renderer that reported one as the other would misstate how
    #: much data an arm rests on.
    n_population_rows: int
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
            "n_population_rows": self.n_population_rows,
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
    #: ``"{dataset element}|{split element}"`` -> ``declaration``, ``describe``, ``recurrence``.
    #: Facts about the data the run measured, recorded once per (dataset, split) so that a
    #: renderer does not have to rebuild the dataset to describe it. ``recurrence`` is
    #: ``None`` for a dataset that declares no coverage, because the statistic is not
    #: defined without it.
    dataset_stats: dict[str, dict] = field(default_factory=dict)
    #: ``"{dataset element}|{population}"`` -> ``{"changes": rows in the window,
    #: "faults": rows averaged over}``. Keyed by the *pair* rather than by the population
    #: name alone: a run over two datasets can have a population of the same name and
    #: different sizes in each, and reporting the first dataset's counts for the second is
    #: the same error as :meth:`describe` guessing which dataset was meant. A single-dataset
    #: run is unaffected, because then there is exactly one key per population name.
    population_sizes: dict[str, dict] = field(default_factory=dict)
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

    def declaration(self, dataset: str | None = None, split: str | None = None) -> dict:
        """The dataset's own declaration, which a run records once per dataset."""
        return self._stats(dataset, split)["declaration"]

    def recurrence(self, dataset: str | None = None, split: str | None = None):
        """How often ``(file, test)`` pairs recur, or ``None`` if coverage is not declared."""
        return self._stats(dataset, split)["recurrence"]

    def population_size(
        self, population: str, dataset: str | None = None
    ) -> tuple[int, int]:
        """``(rows in the evaluation window, rows the metric averaged over)``.

        The second is the fault-bearing subset, so it is what every recall in the report was
        averaged over. Both come from the cells rather than from re-deriving the population,
        so a renderer cannot report a size the run did not use.

        ``dataset`` names the dataset element, and is only needed when the run had more than
        one -- the same rule :meth:`describe` follows. A report that recorded this population
        for several datasets refuses to guess, because returning the first one's counts is
        indistinguishable from a correct answer.
        """
        if dataset is None:
            keys = sorted(
                key for key in self.population_sizes if key.endswith(f"|{population}")
            )
            if not keys:
                raise KeyError(
                    f"this report records no sizes for population {population!r}; it has "
                    f"{sorted(self.population_sizes)}"
                )
            if len(keys) > 1:
                raise KeyError(
                    f"this report records population {population!r} for several datasets "
                    f"{[key.split('|', 1)[0] for key in keys]}; name which one"
                )
            entry = self.population_sizes[keys[0]]
        else:
            key = f"{dataset}|{population}"
            if key not in self.population_sizes:
                raise KeyError(
                    f"this report records no population {population!r} for dataset "
                    f"{dataset!r}; it has {sorted(self.population_sizes)}"
                )
            entry = self.population_sizes[key]
        return entry["changes"], entry["faults"]

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
            "dataset_stats": self.dataset_stats,
            "population_sizes": self.population_sizes,
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
