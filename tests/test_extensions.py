"""Tests for the extension points -- the reason the modules were split at all.

The critique that motivated this restructure was that adding a feature meant editing a
central assembly function, and adding a subset meant editing a module-level dict in
the contract. These tests assert the opposite, and they are written so that the *only*
files changed to add a feature or a subset are the ones here.
"""

from __future__ import annotations

import numpy as np
import pytest

from rts import features
from rts.data import accessors, contract, reporting, splits, subsets
from rts.data.contract import Capability, Requirement, Undefined
from rts.features.block import FeatureBlock, FeatureColumn, FeatureGroup
from tests.stub_dataset import StubDataset

# --- adding a feature ------------------------------------------------------


def _name_lengths(inputs) -> tuple[np.ndarray]:
    """A new column, computed from inputs the contract already provides."""
    values = np.array(
        [len(t.split("::")[-1]) for t in inputs["test_ids"]], dtype=np.float32
    )
    return (np.broadcast_to(values[None, :], (len(inputs["changes"]), len(values))),)


EXTENDED = FeatureBlock(
    name="extended",
    groups=features.STRUCTURED.groups
    + (
        FeatureGroup(
            columns=(FeatureColumn("test_name_length", "custom"),),
            needs=("changes", "test_ids"),
            produce=_name_lengths,
            note="the test function's name length; added without touching rts/",
        ),
    ),
)


def test_a_new_feature_is_one_group_in_one_place():
    ds = StubDataset()
    matrix = features.structured(ds, block=EXTENDED)
    assert matrix.columns[-1] == "test_name_length"
    assert matrix.column("test_name_length")[0].tolist() == [len("test_one"), len("test_two"), len("test_three")]
    # The existing columns are untouched by the addition.
    base = features.structured(ds)
    assert matrix.keep(base.columns).shape == base.X.shape
    assert np.array_equal(matrix.keep(base.columns), base.X)


def test_a_new_family_can_be_withheld_like_a_builtin_one():
    ds = StubDataset()
    withheld = features.structured(
        ds, block=EXTENDED.without_families("custom")
    )
    assert withheld.column("test_name_length").max() == 0.0
    assert withheld.is_undefined("test_name_length")
    # Withholding and genuine absence take the same path, so both report "withheld" vs a
    # named requirement -- the two are distinguishable.
    assert withheld.reason("test_name_length").requirement == "withheld"
    assert "custom" in EXTENDED.families


def test_a_group_whose_requirements_are_unmet_is_unmeasured_not_an_error():
    """A custom group that needs coverage on a dataset without it."""
    def counts(inputs):
        values = np.array(
            [len(c) for c in inputs["coverage"]], dtype=np.float32
        )
        return (np.broadcast_to(values[:, None], (len(values), len(inputs["test_ids"]))),)

    block = FeatureBlock(
        name="needs-coverage",
        groups=(
            FeatureGroup(
                columns=(FeatureColumn("n_covered", "custom"),),
                needs=("coverage", "test_ids"),
                produce=counts,
            ),
        ),
    )
    matrix = features.structured(StubDataset(coverage=False), block=block)
    assert matrix.is_undefined("n_covered")
    assert matrix.reason("n_covered").requirement == Requirement.COVERAGE.value
    assert matrix.column("n_covered").max() == 0.0
    # The same block on a dataset that has coverage produces real values, so the
    # undefined result above is about the dataset and not about the group.
    with_coverage = features.structured(StubDataset(), block=block)
    assert with_coverage.column("n_covered")[0].max() == 2.0


def test_material_a_group_names_but_the_caller_does_not_supply_raises():
    """Caller-supplied inputs is a promise, so failing to supply it is a bug, not a zero."""
    block = FeatureBlock(
        name="external",
        groups=(
            FeatureGroup(
                columns=(FeatureColumn("from_outside", "custom"),),
                needs=("something_external",),
                produce=lambda m: (m["something_external"],),
            ),
        ),
    )
    assert block.external_inputs == frozenset({"something_external"})
    with pytest.raises(ValueError, match="caller-supplied inputs"):
        block.build(StubDataset())
    supplied = np.ones((4, 3), dtype=np.float32)
    matrix = block.build(StubDataset(), extra={"something_external": supplied})
    assert np.array_equal(matrix.column("from_outside"), supplied)


def test_an_unknown_column_name_fails_loudly():
    matrix = features.structured(StubDataset())
    with pytest.raises(KeyError, match="no column"):
        matrix.index("nope")
    with pytest.raises(KeyError, match="no column"):
        features.STRUCTURED.index("filename_stem_match_typo")
    with pytest.raises(KeyError, match="no column"):
        features.STRUCTURED.without("nope")


def test_the_two_blocks_agree_on_their_shared_columns():
    """The bundle block reads the base's columns by name, so a rename is caught."""
    from rts.features.bundle import BASE_COLUMNS

    for name in BASE_COLUMNS:
        assert name in features.STRUCTURED.columns


def test_the_traceability_family_is_exactly_its_three_real_columns():
    """Regression: the family used to name ``n_tests_in_file``, which is not a column.

    A name-based ablation ignores a name that matches nothing, so the L3 rung silently
    kept ``tests_per_file`` -- real traceability information -- while reporting
    itself as traceability-free. Withholding is strict now, so the same typo raises.
    """
    assert features.STRUCTURED.family("proximity") == (
        "path_proximity",
        "tests_per_file",
        "filename_match",
    )
    assert "n_tests_in_file" not in features.STRUCTURED.columns
    with pytest.raises(KeyError):
        features.STRUCTURED.without("n_tests_in_file")
    # And asking to withhold a family that does not exist is also an error, so the typo
    # that caused the leak cannot be expressed at all.
    with pytest.raises(KeyError):
        features.STRUCTURED.without_families("n_tests_in_file")


def test_every_ladder_rung_withholds_exactly_the_columns_its_families_name():
    """A rung must withhold a real, non-empty set -- that is the whole ablation."""
    from rts import studies

    for rung, removed in studies.RUNGS:
        block = studies.rung_block(removed)
        expected = {
            column for family in removed for column in features.STRUCTURED.family(family)
        }
        assert set(block.suppressed) == expected, rung
        if removed:
            assert expected, rung  # a rung that removes nothing is not a rung
    # The cumulative rungs are nested, and L3 is the one that must leave no
    # traceability column standing.
    l3 = studies.rung_block(studies.RUNGS[-1][1])
    assert set(l3.suppressed) == set(
        features.STRUCTURED.family("temporal")
        + features.STRUCTURED.family("coverage")
        + features.STRUCTURED.family("proximity")
    )
    assert len(l3.suppressed) == 9


# --- adding a subset ---------------------------------------------------


def _long_named_tests(inputs) -> np.ndarray:
    return np.array(
        [any(len(t.split("::")[-1]) > 8 for t in tests) for tests in inputs["killing"]],
        dtype=bool,
    )


LONG_NAMES = subsets.Subset(
    name="long_test_names",
    note="a subset added here, not in rts/data/subsets.py",
    needs=("killing",),
    predicate=_long_named_tests,
)


MINE = subsets.SubsetRegistry(
    subsets.STUDY.subsets + (LONG_NAMES,)
)


def test_a_new_population_is_a_value_the_caller_supplies():
    assert "long_test_names" not in subsets.STUDY.names()
    assert LONG_NAMES in MINE.subsets
    assert MINE.get("long_test_names") is LONG_NAMES
    # The study's own set is unchanged by the caller's addition.
    assert subsets.STUDY.names() == (
        "detectable",
        "no_prior_failure",
        "cold_start",
        "low_cooccurrence",
    )


def test_a_new_populations_requirements_are_derived_not_written():
    # It reads only "killing", which is always available, so it needs nothing beyond labels.
    assert LONG_NAMES.requires() == frozenset({Requirement.LABELS})
    assert LONG_NAMES.unavailable(StubDataset()) is None
    mask = LONG_NAMES.mask(StubDataset())
    assert mask.tolist() == [False, False, True, False]


def test_withholding_a_family_that_does_not_exist_is_an_error():
    """A typo in an ablation must not silently ablate nothing."""
    assert not EXTENDED.has_family("nope")
    with pytest.raises(KeyError):
        EXTENDED.without_families("hitory")  # deliberate typo
    # But withholding a family that IS present, when the block may or may not have it,
    # is guarded by asking first -- which is what features.structured does for history.
    assert EXTENDED.has_family("temporal")
    custom_only = FeatureBlock(
        name="custom-only",
        groups=(EXTENDED.groups[-1],),
    )
    assert not custom_only.has_family("temporal")


def test_keeping_part_of_a_group_projects_its_computation():
    """A group computes its columns together, so keeping one must project, not re-derive."""
    ds = StubDataset()
    full = features.structured(ds)
    # Two of the three coverage columns, taken from the middle and end of one group.
    partial = features.STRUCTURED.only(["function_coverage", "coverage_set_size_prior"])
    matrix = features.structured(ds, block=partial)
    assert matrix.columns == ("function_coverage", "coverage_set_size_prior")
    assert np.array_equal(matrix.column("function_coverage"), full.column("function_coverage"))
    assert np.array_equal(
        matrix.column("coverage_set_size_prior"), full.column("coverage_set_size_prior")
    )
    with pytest.raises(KeyError):
        features.STRUCTURED.only(["function_coverage", "nope"])


def test_a_population_over_coverage_material_is_unavailable_without_it():
    over_coverage = subsets.Subset(
        name="over_coverage",
        note="requires coverage because it reads coverage inputs",
        needs=("coverage",),
        predicate=lambda inputs: np.array(
            [bool(tests) for tests in inputs["coverage"]], dtype=bool
        ),
    )
    # The requirement was never written down; it came from the inputs name.
    assert over_coverage.requires() == frozenset({Requirement.COVERAGE})
    assert over_coverage.unavailable(StubDataset()) is None
    missing = over_coverage.unavailable(StubDataset(coverage=False))
    assert isinstance(missing, Undefined) and missing.requirement == "coverage"


def test_extending_the_registry_does_not_mutate_the_original():
    extra = subsets.SubsetRegistry((LONG_NAMES,))
    combined = subsets.STUDY + extra
    assert combined.names()[-1] == "long_test_names"
    assert "long_test_names" not in subsets.STUDY.names()
    # Composition is left-biased, so re-adding an existing name does not duplicate it.
    assert (combined + extra).names() == combined.names()


def test_a_registry_resolves_names_and_rejects_unknown_ones():
    assert subsets.subset("cold_start", MINE) is subsets.COLD_START
    with pytest.raises(KeyError):
        subsets.subset("nope", MINE)
    assert subsets.resolve(LONG_NAMES, MINE) is LONG_NAMES
    assert subsets.resolve("detectable", MINE) is subsets.DETECTABLE


def test_a_population_can_be_evaluated_against_any_dataset():
    from rts import evaluate

    ds = StubDataset()
    # Half the changes are held out, so change 2 -- the only long-named killing test -- is
    # inside the evaluation window.
    split = splits.make_split(ds, train_fraction=0.5)
    scores = accessors.labels(ds).astype(np.float32)
    evaluation = evaluate.evaluate_rows(
        scores, ds, split.test_idx, budgets=(0.5,), n_bootstrap=0, subset=LONG_NAMES
    )
    assert evaluation.subset == "long_test_names"
    assert evaluation.measured
    assert evaluation.n_rows == 1


def test_a_populations_declaration_is_reportable():
    assert LONG_NAMES.metadata() == {
        "name": "long_test_names",
        "needs": ["killing"],
        "requires": ["labels"],
        "note": "a subset added here, not in rts/data/subsets.py",
    }
    assert len(subsets.STUDY.describe()) == 4
    assert any(w.code == "dataset.multi_file_changes_flattened" for w in reporting.audit(
        StubDataset(extra_files={"c0": ("pkg/other.py",)})
    ))


# --- adding a capability ---------------------------------------------------


def test_an_unknown_capability_is_rejected_rather_than_ignored():
    """A dataset cannot invent a capability: the vocabulary is closed and checked."""

    class Bogus(StubDataset):
        def capabilities(self):
            return frozenset({Capability.COVERAGE, "not_a_capability"})  # type: ignore[arg-type]

    ranged = Bogus()
    assert ranged.has_capability(Capability.COVERAGE)
    with pytest.raises(ValueError):
        ranged.has_capability("not_a_capability")


def test_adding_a_capability_is_a_vocabulary_change_in_one_place():
    """Extending the capability vocabulary is one enum member, and nothing else guesses."""
    assert {c.value for c in Capability} == {"coverage", "durations"}
    for capability in Capability:
        assert contract.requirement_for(capability).value == capability.value
    # The inputs catalogue is the single source of what reading something requires, so a
    # capability and its requirement cannot drift apart.
    assert accessors.INPUTS["coverage"].requires == frozenset({Requirement.COVERAGE})
    assert accessors.INPUTS["durations"].requires == frozenset({Requirement.DURATIONS})
    assert accessors.INPUTS["labels"].requires == frozenset({Requirement.LABELS})
    assert accessors.INPUTS["changes"].requires == frozenset()
    assert accessors.INPUTS["pair_failures"].requires == frozenset({Requirement.LABELS})
    assert accessors.INPUTS["pairs"].requires == frozenset({Requirement.COVERAGE})


def test_requirement_for_maps_a_capability_to_its_requirement():
    assert contract.requirement_for(Capability.COVERAGE) is Requirement.COVERAGE
    assert contract.requirement_for(Capability.DURATIONS) is Requirement.DURATIONS
