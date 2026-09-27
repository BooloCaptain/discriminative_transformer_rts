"""The BugsInPy arm: the study's only real-label data, declared as a sweep.

Every other arm is measured on labels mutmut produced by running only the tests that cover the
mutated function, which makes the coverage feature circular with respect to them. BugsInPy's
failing tests come from the projects' own bug reports, so no feature is circular -- and the
corpus has no coverage and no durations at all, which makes it structurally the ladder's hardest
rung (L3: no coverage, no traceability, no history) on real data rather than on a synthetic SUT.

What is different about this arm, in layer terms
------------------------------------------------
* **One dataset, eight populations.** The projects are pooled for the headline number, and each
  project is a population, so the per-project breakdown is the same machinery as the pooled
  number rather than a second loop.
* **The candidates are the change's own pool** (``candidates="own"``). A budget is a fraction of
  *that bug's* project suite, not of the union of eight, which would make the budget mean eight
  different things.
* **There is no training stage.** A zero-shot text model has nothing to fit, so the split holds
  nothing out: ``train_fraction=0.0`` puts every bug in the evaluation window and leaves the
  train prefix empty. That is a declaration, not an accident.
* **One budget per run.** The recorded intervals came from a fresh bootstrap generator per
  budget, because the original looped budgets calling ``evaluate`` one at a time. Sweeping the
  four in one call would consume one generator across them and move every interval, and
  ``budgets`` is a knob -- so this is four runs sharing one score cache.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .. import config
from ..data import composition, datasets, populations
from ..data.contract import Dataset
from ..experiment import (
    ROLE_DATASET,
    ROLE_MODEL,
    ROLE_POPULATION,
    Axis,
    Comparison,
    Element,
    Experiment,
    Knobs,
    RunReport,
    constant,
    run,
)
from ..model import selectors
from .axes import _model_element, split_axis, structured_feature_axis

#: The arm's SemIf cache. Keyed by ``(project/bug_id, column-in-that-bug's-pool)`` rather than
#: by ``(change_id, test_nodeid)``, which is why it needs its own loader: the column is a
#: position in the change's own pool, and the pool is what the budget is a fraction of.
SEMIF_CACHE = config.ARTIFACTS / "semif_scores_bugsinpy.jsonl"

#: The budget grid, and the two resample counts. Different counts for the table intervals and
#: the paired tests, as in the other arms: two quantities, two precision needs.
BUGSINPY_BUDGETS: tuple[float, ...] = (0.01, 0.05, 0.1, 0.2)
TABLE_RESAMPLES = 1000
PAIRED_RESAMPLES = 2000

#: The model order the recorded artifact's tables are written in. Declared rather than read back
#: out of the report, so a reordered axis cannot silently reorder a table.
MODEL_ORDER: tuple[str, ...] = ("random", "bm25_lexical", "semif_reranker")


def load_scores(path: Path, ds: Dataset) -> np.ndarray:
    """Read the BugsInPy SemIf cache into a ``[n_changes, n_tests]`` matrix.

    The column in each record is a position within that bug's pool, so it is resolved through
    ``Dataset.own_candidate_pool`` rather than through a module-level map. That is what lets the
    file be read by a *value* -- the pooled dataset -- instead of by a driver, and it means the
    pool's declared order is load-bearing: if it were wrong, the scores would land on the wrong
    tests and the recorded numbers would move.
    """
    pool = ds.own_candidate_pool()
    if pool is None:
        raise ValueError(
            f"dataset {ds.name!r} declares no per-change pool, so {path.name} cannot be read"
        )
    rows = {}
    for index, change in enumerate(ds.changes):
        rows[ds.change_id(change).replace("::", "/", 1)] = index

    out = np.full((ds.n_changes, ds.n_tests), -1e9, dtype=np.float32)
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            row = rows.get(record["bug"])
            if row is None:
                continue
            cols = np.flatnonzero(pool[row])
            if 0 <= record["col"] < cols.size:
                out[row, cols[record["col"]]] = record["score"]
    return out


# --- the axes ---------------------------------------------------------------


def dataset_axis(projects: Sequence[str] | None = None) -> Axis:
    return Axis(
        ROLE_DATASET,
        (
            Element(
                "bugsinpy_pooled",
                lambda _binding, projects=projects: pooled_dataset(projects),
                tier="cpu",
                note="the selected projects as one evaluation population, test ids namespaced",
            ),
        ),
        note="the real-label corpus: one dataset, pooled, one population per project",
    )


def model_axis() -> Axis:
    """The three selectors this arm can afford. No tree: it has no features to fit on.

    Random and BM25 are the *per-change* forms, because a global BM25 index over eight projects
    would let one project's vocabulary move another's scores, and a global random draw would
    give each bug a different set of ranks.
    """
    return Axis(
        ROLE_MODEL,
        (
            _model_element(
                selectors.PerPoolRandomSelector(candidates_mode="own"),
                note="uniform, drawn per bug over that bug's own pool",
            ),
            _model_element(
                selectors.PerPoolLexicalSelector(candidates_mode="own", query="diff"),
                note=(
                    "BM25 fitted per bug over its own pool; the query is the raw diff, which "
                    "is what this arm's recorded numbers used"
                ),
            ),
            Element(
                "semif_reranker",
                lambda _binding: selectors.CachedScores(
                    "semif_reranker", SEMIF_CACHE, loader=load_scores
                ),
                tier="cpu",
                note=f"precomputed across all pairs, from {SEMIF_CACHE.name}",
            ),
        ),
        note="the zero-shot and cheap-classical selectors, per-change scope",
    )


def project_names(projects: Sequence[str] | None = None) -> tuple[str, ...]:
    """The projects a run covers, in sorted order: all built ones, or the named subset.

    Declared here rather than in the driver because the *arm* has to know: a corpus of three
    projects is a different experiment from a corpus of eight, so the selection is part of the
    dataset element rather than a filter applied to its results.
    """
    available = tuple(datasets.available_bugsinpy_projects())
    if projects is None:
        return available
    missing = sorted(set(projects) - set(available))
    if missing:
        raise FileNotFoundError(f"no built dataset for project(s): {missing}")
    wanted = set(projects)
    return tuple(name for name in available if name in wanted)


def pooled_dataset(projects: Sequence[str] | None = None) -> Dataset:
    """The selected projects as one evaluation population, namespacing test ids."""
    return composition.pool(
        [datasets.bugsinpy(name) for name in project_names(projects)], name="bugsinpy"
    )


def project_populations(ds: Dataset) -> dict[str, populations.Population]:
    """One population per project, from the namespaced change ids.

    A population names the material its predicate reads; this one reads none, because the mask
    is a fact about *which project a row came from* and is therefore precomputed from the
    dataset. Declaring the mask as material would be a second, weaker way to say the same thing.
    """
    ids = [ds.change_id(change) for change in ds.changes]
    out: dict[str, populations.Population] = {}
    for name in sorted({value.split("::", 1)[0] for value in ids}):
        mask = np.array([value.startswith(f"{name}::") for value in ids], dtype=bool)
        out[name] = populations.Population(
            name=name,
            note=f"the {name} project's bugs, inside the pooled corpus",
            needs=(),
            predicate=lambda _material, mask=mask: mask,
        )
    return out


def population_axis(projects: Sequence[str] | None = None) -> Axis:
    """The pooled population and one population per project.

    The per-project breakdown is the reason this arm needs a population axis at all: it is the
    check that "SemIf ties BM25 on the pooled corpus" is either a claim that holds everywhere or
    one carried by a single project.
    """
    elements: list[Element] = [
        constant(
            "fault_bearing",
            populations.FAULT_BEARING,
            note="every bug, since every bug in this corpus has a failing test",
        )
    ]
    elements.extend(
        Element(
            name,
            lambda binding, name=name: project_populations(binding.dataset)[name],
            tier="cpu",
            note=f"the {name} project's bugs",
        )
        for name in project_names(projects)
    )
    return Axis(ROLE_POPULATION, tuple(elements), note="the corpus, and each project in it")


# --- the arm ----------------------------------------------------------------


def arm(
    budget: float = 0.05,
    *,
    projects: Sequence[str] | None = None,
    name: str | None = None,
) -> Experiment:
    """The arm at one budget, because a budget is a knob and the intervals are per-budget."""
    return Experiment(
        name=name or f"bugsinpy.b{budget:.2f}",
        datasets=dataset_axis(projects),
        features=structured_feature_axis(),
        models=model_axis(),
        populations=population_axis(projects),
        # A zero-train split: this arm has no training stage, so the window is every bug.
        splits=split_axis(train_fraction=0.0),
        knobs=Knobs(
            seed=config.SEED,
            budgets=(budget,),
            n_bootstrap=TABLE_RESAMPLES,
            n_bootstrap_paired=PAIRED_RESAMPLES,
            candidates="own",
        ),
        # References are the baselines because the *cell* is the one paired and the recorded
        # convention is "SemIf minus baseline", positive when SemIf is better.
        comparisons=tuple(
            Comparison(ROLE_MODEL, reference, budget)
            for reference in ("bm25_lexical", "random")
        ),
        # Explicitly off, and it must stay off: the corpus has no execution history, and a
        # future split that shuffled would otherwise switch the history family on silently.
        history=False,
        note="the BugsInPy arm: real labels, no coverage, no history",
    )


def run_arm(
    budgets: tuple[float, ...] = BUGSINPY_BUDGETS,
    *,
    projects: Sequence[str] | None = None,
    out_dir: Path | str | None = None,
    save: bool = True,
    verbose: bool = True,
) -> dict[float, RunReport]:
    """Run one experiment per budget, sharing **one** score cache between them.

    Sharing is sound for the same reason it is in ``run_study``: a score matrix is a function of
    the cell's context, and the population is not in that key, so the four budgets cost one set
    of scores and four metric sweeps.
    """
    scores: dict = {}
    reports: dict[float, RunReport] = {}
    for budget in budgets:
        reports[budget] = run(
            arm(budget, projects=projects),
            out_dir=out_dir,
            scores=scores,
            save=save,
            verbose=verbose,
        )
    return reports


#: Flat-namespace aliases, so ``studies.bugsinpy_arm`` reads like the study's other arms.
bugsinpy_arm = arm
run_bugsinpy = run_arm
