"""Tests for the BugsInPy arm: the real labels, and the semantics of a pooled corpus.

These need the built per-project datasets under ``artifacts/bugsinpy/`` (no checkout, no test
execution, no GPU), and they load them once per module. What is pinned is the contract *between*
the arm's declarations and the recorded artifact, plus two invariants that would make the arm
meaningless if they broke: the candidate pool must contain every failing test, and a bug must not
be rankable against another project's tests.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from rts import bugsinpy, config, models, studies
from rts.data import accessors

# Aliased so pytest does not try to collect the enum as a test class.
from rts.data.contract import TestUnit as Unit


@pytest.fixture(scope="module")
def corpus():
    projects = bugsinpy.load_datasets()
    pooled = bugsinpy.pooled_dataset(projects)
    return projects, pooled, accessors.candidates(pooled, "own")


def _recorded() -> dict:
    return json.loads((config.ARTIFACTS / "bugsinpy_results.json").read_text())


# --- the pool, which is what a budget is a fraction of ----------------------


def test_the_pool_contains_every_bugs_failing_tests(corpus):
    """If a killer were not a candidate, that bug could not be caught at any budget.

    The arithmetic analogue of the marshmallow invariant that no killing test falls outside the
    coverage set: it holds by construction here too, and it is the reason a per-bug budget means
    something.
    """
    projects, pooled, pool = corpus
    index = pooled.test_index
    for ds in projects:
        for bug in ds.changes:
            row = accessors.change_index(pooled)[f"{ds.name}::{ds.change_id(bug)}"]
            for nodeid in bug.failing:
                col = index[pooled.canonical_test_id(f"{ds.name}::{nodeid}")]
                assert pool[row, col], f"{ds.name}/{ds.change_id(bug)} cannot catch {nodeid}"


def test_a_bug_is_only_rankable_within_its_own_project(corpus):
    """A budget of a fraction of the union of eight suites would mean eight different things."""
    projects, pooled, pool = corpus
    assert pool.shape == (pooled.n_changes, pooled.n_tests)
    # No project is a large majority of the corpus, so a wrong layout would show up immediately.
    assert 0 < pool.sum() < pool.size
    per_project = [pool[accessors.change_index(pooled)[f"{ds.name}::{ds.change_id(b)}"]].sum()
                   for ds in projects for b in ds.changes]
    assert min(per_project) > 1
    assert max(per_project) < pooled.n_tests


def test_the_pool_reproduces_the_legacy_candidate_matrix(corpus):
    """The declaration is the value the arm was always measured against, not a re-description.

    ``bugsinpy.candidate_matrix`` built this mask inline before the contraction moved it onto the
    dataset. Comparing against it is the only check that the layout (rows per project, columns
    per namespaced test id) survived.
    """
    projects, pooled, pool = corpus
    index = pooled.test_index
    legacy = np.zeros_like(pool)
    for ds in projects:
        for bug in ds.changes:
            row = accessors.change_index(pooled)[f"{ds.name}::{ds.change_id(bug)}"]
            for nodeid in bug.pool:
                col = index.get(f"{ds.name}::{nodeid}")
                if col is not None:
                    legacy[row, col] = True
    assert np.array_equal(pool, legacy)


# --- the semantics of a pool -------------------------------------------------


def test_the_projects_agree_on_what_a_test_is_and_the_declaration_says_so(corpus):
    """A pool flattens ``test_unit``; when the constituents disagree, that has to be recorded.

    Here they agree (every project enumerates test *cases*), so the pool must not emit the
    ``pool.mixed_test_unit`` note -- a warning that fired unconditionally would be noise, and one
    that never fired would leave the flattening unstated.
    """
    projects, pooled, _ = corpus
    assert {ds.test_unit() for ds in projects} == {Unit.CASE}
    assert pooled.test_unit() is Unit.CASE
    assert pooled.mixed_test_units() == ()
    assert [w.code for w in pooled.integrity_notes()] == []


def test_the_arm_has_no_coverage_or_history_to_trust(corpus):
    """The arm's whole claim: no feature is circular with respect to the labels."""
    _, pooled, _ = corpus
    assert pooled.capabilities() == frozenset()
    assert pooled.ordering().value == "imposed"
    # And the arm must not silently acquire history features from a future shuffling split.
    assert studies.bugsinpy_arm(0.05).history is False


# --- the arm against the recorded artifact ----------------------------------


def test_the_arm_declares_the_recorded_selector_keys():
    assert list(studies.bugsinpy_arm(0.05).models.names()) == list(
        _recorded()["results"]
    )
    assert studies.bugsinpy.MODEL_ORDER == tuple(_recorded()["results"])


def test_the_arm_populations_are_the_recorded_projects():
    names = set(studies.bugsinpy_arm(0.05).populations.names())
    assert names == {"fault_bearing"} | set(_recorded()["per_project_recall_at_0.05"])


def test_the_arm_pairs_at_every_recorded_budget():
    """The recorded keys name the pairing (``semif_vs_x``); the arm names the reference.

    The *cell* is SemIf, because the recorded convention is "SemIf minus baseline" -- positive
    when SemIf is better -- and ``_compare`` computes ``cell - reference``.
    """
    recorded = _recorded()["comparisons"]
    for budget in studies.BUGSINPY_BUDGETS:
        arm = studies.bugsinpy_arm(budget)
        assert list(arm.knobs.budgets) == [budget]
        assert {c.reference for c in arm.comparisons} == {"bm25_lexical", "random"}
        assert set(recorded[f"{budget:.2f}"]) == {"semif_vs_bm25", "semif_vs_random"}


def test_the_arm_holds_nothing_out(corpus):
    """A zero-shot model has nothing to fit, so the split is every bug with an empty prefix."""
    _, pooled, _ = corpus
    arms = studies.bugsinpy_arm(0.05)
    assert arms.splits.names() == ("split0",)
    from rts.data import splits

    split = splits.make_split(pooled, train_fraction=0.0)
    assert split.train_idx.size == 0
    assert len(split.test_idx) == pooled.n_changes


def test_a_project_selection_is_a_different_corpus():
    """A corpus of one project is a different experiment, not a filter on the results.

    The selection has to reach the *dataset element*, or a subset run would audit one project
    while measuring all eight -- and that is silent, because every number produced would still
    be a number.
    """
    from rts.experiment import ROLE_DATASET

    arm = studies.bugsinpy_arm(0.05, projects=["black"])
    assert set(arm.populations.names()) == {"fault_bearing", "black"}
    built = studies.bugsinpy.pooled_dataset(["black"])
    assert built.n_changes == 19
    assert built.n_tests == 145
    assert studies.bugsinpy.project_names(["black"]) == ("black",)
    assert arm.axes()[ROLE_DATASET].names() == ("bugsinpy_pooled",)


def test_an_unknown_project_is_an_error_not_an_empty_corpus():
    with pytest.raises(FileNotFoundError, match="no built dataset"):
        studies.bugsinpy.project_names(["not_a_project"])


def test_the_cache_loader_resolves_columns_through_each_bugs_pool(corpus):
    """The cache stores a *position* in a bug's pool, so the pool's order is load-bearing."""
    _, pooled, _ = corpus
    matrix = studies.bugsinpy.load_scores(studies.bugsinpy.SEMIF_CACHE, pooled)
    assert matrix.shape == (pooled.n_changes, pooled.n_tests)
    scored = int((matrix > -1e8).sum())
    # Every candidate pair is cached; nothing outside a pool is.
    assert scored == int(accessors.candidates(pooled, "own").sum())


# --- the per-change-scope selectors -----------------------------------------


def _context(pooled):
    from rts import models
    from rts.data import splits

    return models.Context(
        ds=pooled,
        features=None,
        split=splits.make_split(pooled, train_fraction=0.0),
        bm25=None,
        seed=config.SEED,
    )


def test_a_per_pool_random_baseline_only_ranks_within_a_pool(corpus):
    """Outside a change's pool the score is the sentinel, so a bug cannot be credited with a
    test from another project -- which is what makes the baseline comparable across projects."""
    _, pooled, pool = corpus
    scores = models.PerPoolRandomSelector(candidates_mode="own").scores(_context(pooled))
    inside = scores[pool]
    assert ((inside >= 0.0) & (inside < 1.0)).all()
    assert (scores[~pool] < -1e8).all()


def test_a_per_pool_random_baseline_is_reproducible(corpus):
    """The draw sequence is part of a recorded baseline, so it may not be re-derived per row."""
    _, pooled, _ = corpus
    selector = models.PerPoolRandomSelector(candidates_mode="own")
    first = selector.scores(_context(pooled))
    second = models.PerPoolRandomSelector(candidates_mode="own").scores(_context(pooled))
    assert np.array_equal(first, second)


def test_the_bm25_query_choice_is_recorded_rather_than_assumed(corpus):
    """The raw diff and the extracted change lines are different queries, and this arm's
    recorded numbers used the raw diff. The parameter is what makes that reviewable."""
    _, pooled, _ = corpus
    diff = models.PerPoolLexicalSelector(candidates_mode="own", query="diff").scores(
        _context(pooled)
    )
    change = models.PerPoolLexicalSelector(candidates_mode="own", query="change").scores(
        _context(pooled)
    )
    assert not np.array_equal(diff, change)
    with pytest.raises(ValueError, match="query must be"):
        models.PerPoolLexicalSelector(query="whatever")
