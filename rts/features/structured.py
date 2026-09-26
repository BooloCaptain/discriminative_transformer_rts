"""The structured block: the 15 columns the classical selectors read.

The declaration is the whole point of this module. Adding a column means adding it to a
group here -- one file, one line -- rather than editing an assembly function that also
had to know about capability gating, history policy and column order.

Column order is preserved exactly as the recorded artifacts have it, because
``actions`` and figures index columns positionally through
:class:`FeatureMatrix` and the ladder's rung definitions name families. The order is
declared by the group order, so it is visible rather than implied by control flow.

Families are declared per *column*, not per group, because they cut across computations:
the ``traceability`` family takes one column each from two different groups, and the
ladder ablates by family.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..contract import Dataset, Ordering, Policy, Warnings
from . import derived
from .block import FeatureBlock, FeatureColumn, FeatureGroup, FeatureMatrix

#: Families the traceability ladder ablates as a group.
FAMILIES = ("coverage", "traceability", "history")


def _test_index(material: Mapping[str, Any]) -> dict[str, int]:
    return {t: i for i, t in enumerate(material["test_ids"])}


def _coverage_group(material: Mapping[str, Any]) -> Sequence[np.ndarray]:
    return derived.coverage_columns(
        material["coverage"],
        _test_index(material),
        len(material["changes"]),
        len(material["test_ids"]),
    )


def _proximity_group(material: Mapping[str, Any]) -> Sequence[np.ndarray]:
    paths, test_ids = material["paths"], material["test_ids"]
    return (
        derived.path_distance(paths, test_ids),
        derived.n_tests_in_test_file(test_ids),
        derived.filename_stem_match(paths, test_ids),
    )


def _duration_group(material: Mapping[str, Any]) -> Sequence[np.ndarray]:
    return (derived.durations_column(material["durations"], material["test_ids"]),)


def _test_size_group(material: Mapping[str, Any]) -> Sequence[np.ndarray]:
    test_ids = material["test_ids"]
    sources = {t: material["test_source"](t) for t in test_ids}
    return (
        derived.test_n_lines_row(test_ids, sources),
        derived.test_n_tokens_row(test_ids, sources),
    )


def _change_size_group(material: Mapping[str, Any]) -> Sequence[np.ndarray]:
    diffs = [material["diff"](c) for c in material["changes"]]
    n_c, n_t = len(diffs), len(material["test_ids"])
    size = np.array([derived.change_size_in(d) for d in diffs], dtype=np.float32)
    added = np.array([len(derived.changed_lines_in(d)) for d in diffs], dtype=np.float32)
    removed = np.array([len(derived.removed_lines_in(d)) for d in diffs], dtype=np.float32)
    return tuple(
        np.broadcast_to(v[:, None], (n_c, n_t)) for v in (size, added, removed)
    )


def _history_group(material: Mapping[str, Any]) -> Sequence[np.ndarray]:
    return derived.history_features(material["labels"], material["runs"])


#: The structured block. Group order is column order and is load-bearing.
STRUCTURED = FeatureBlock(
    name="structured",
    note="the 15 columns the classical selectors read, in recorded order",
    groups=(
        FeatureGroup(
            columns=(
                FeatureColumn("covers_function", "coverage"),
                FeatureColumn("n_covering_tests", "coverage"),
                FeatureColumn("coverage_rank_prior", "coverage"),
            ),
            needs=("coverage", "changes", "test_ids"),
            produce=_coverage_group,
            note="whether the change's code is observable by the test, and how far",
        ),
        FeatureGroup(
            columns=(
                FeatureColumn("path_distance", "traceability"),
                FeatureColumn("n_tests_in_test_file", "traceability"),
                FeatureColumn("filename_stem_match", "traceability"),
            ),
            needs=("diff", "test_ids"),
            produce=_proximity_group,
            note="co-location and naming, which assume tests live beside the code",
        ),
        FeatureGroup(
            columns=(FeatureColumn("test_duration", "intrinsic"),),
            needs=("durations", "test_ids"),
            produce=_duration_group,
            note="wall-clock; hardware-dependent, so available but not comparable",
        ),
        FeatureGroup(
            columns=(
                FeatureColumn("test_n_lines", "intrinsic"),
                FeatureColumn("test_n_tokens", "intrinsic"),
            ),
            needs=("test_source", "test_ids"),
            produce=_test_size_group,
            note="test size, from the test's own source",
        ),
        FeatureGroup(
            columns=(
                FeatureColumn("change_size", "intrinsic"),
                FeatureColumn("change_added_lines", "intrinsic"),
                FeatureColumn("change_removed_lines", "intrinsic"),
            ),
            needs=("diff",),
            produce=_change_size_group,
            note="change size, parsed from the unified diff",
        ),
        FeatureGroup(
            columns=(
                FeatureColumn("test_failure_rate_cum", "history"),
                FeatureColumn("test_runs_cum", "history"),
                FeatureColumn("test_last_failure_age", "history"),
            ),
            needs=("labels", "runs"),
            produce=_history_group,
            note="cumulative over strictly earlier changes; needs a real order to mean anything",
        ),
    ),
)


def structured(
    ds: Dataset,
    *,
    history: bool | None = None,
    block: FeatureBlock | None = None,
    warnings: Warnings | None = None,
) -> FeatureMatrix:
    """Build the structured block against ``ds``.

    ``history`` selects the cumulative-history columns. ``None`` means *the default*: on
    for an :attr:`Ordering.OBSERVED` dataset, off for an :attr:`Ordering.IMPOSED` one,
    where a cumulative feature over an arbitrary order manufactures a leak rather than
    merely failing to be interpretable.

    Study entry points that reproduce the documented numbers opt in explicitly, and the
    accompanying warning travels with the returned matrix rather than being stored on
    the dataset -- so the record describes the derivation, not the call history.
    """
    collected = warnings if warnings is not None else Warnings()
    use_history = ds.ordering() is Ordering.OBSERVED if history is None else bool(history)

    if use_history and ds.ordering() is Ordering.IMPOSED:
        collected.add(
            "feature.history_on_imposed_order",
            Policy.OBSERVED_ORDER.value,
            f"history features are enabled on an imposed order (order_seed={ds.order_seed}); "
            "the values are only interpretable under that seed and should be reported as a "
            "spread over seeds, not as one number",
            scope="features:history",
        )

    chosen = block if block is not None else STRUCTURED
    if not use_history and chosen.has_family("history"):
        # Withholding a family the block does not declare is an error, not a no-op, so a
        # custom block without a history family is left alone rather than rejected here.
        chosen = chosen.without_families("history")
    return chosen.build(ds, collected)


__all__ = ["FAMILIES", "STRUCTURED", "structured"]
