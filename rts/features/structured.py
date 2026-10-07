"""The structured block: the 15 columns the classical rankers read.

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

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..data.contract import Dataset, Diagnostics, Ordering, Policy
from . import derived
from .block import FeatureBlock, FeatureColumn, FeatureGroup, FeatureMatrix

#: Families the traceability ladder ablates as a group.
FAMILIES = ("coverage", "proximity", "temporal")


def _test_index(inputs: Mapping[str, Any]) -> dict[str, int]:
    return {t: i for i, t in enumerate(inputs["test_ids"])}


def _coverage_group(inputs: Mapping[str, Any]) -> Sequence[np.ndarray]:
    return derived.coverage_columns(
        inputs["coverage"],
        _test_index(inputs),
        len(inputs["changes"]),
        len(inputs["test_ids"]),
    )


def _proximity_group(inputs: Mapping[str, Any]) -> Sequence[np.ndarray]:
    paths, test_ids = inputs["paths"], inputs["test_ids"]
    return (
        derived.path_proximity(paths, test_ids),
        derived.tests_per_file(test_ids),
        derived.filename_match(paths, test_ids),
    )


def _duration_group(inputs: Mapping[str, Any]) -> Sequence[np.ndarray]:
    return (derived.durations_column(inputs["durations"], inputs["test_ids"]),)


def _test_size_group(inputs: Mapping[str, Any]) -> Sequence[np.ndarray]:
    test_ids = inputs["test_ids"]
    sources = {t: inputs["test_source"](t) for t in test_ids}
    return (
        derived.test_n_lines_row(test_ids, sources),
        derived.test_n_tokens_row(test_ids, sources),
    )


def _change_size_group(inputs: Mapping[str, Any]) -> Sequence[np.ndarray]:
    diffs = [inputs["diff"](c) for c in inputs["changes"]]
    n_c, n_t = len(diffs), len(inputs["test_ids"])
    size = np.array([derived.change_size_in(d) for d in diffs], dtype=np.float32)
    added = np.array([len(derived.changed_lines_in(d)) for d in diffs], dtype=np.float32)
    removed = np.array([len(derived.removed_lines_in(d)) for d in diffs], dtype=np.float32)
    return tuple(
        np.broadcast_to(v[:, None], (n_c, n_t)) for v in (size, added, removed)
    )


def _temporal_group(inputs: Mapping[str, Any]) -> Sequence[np.ndarray]:
    return derived.temporal_features(inputs["labels"], inputs["execution"])


#: The structured block. Group order is column order and is load-bearing.
STRUCTURED = FeatureBlock(
    name="structured",
    note="the 15 columns the classical rankers read, in recorded order",
    groups=(
        FeatureGroup(
            columns=(
                FeatureColumn("function_coverage", "coverage"),
                FeatureColumn("coverage_set_size", "coverage"),
                FeatureColumn("coverage_set_size_prior", "coverage"),
            ),
            needs=("coverage", "changes", "test_ids"),
            produce=_coverage_group,
            note="whether the change's code is observable by the test, and how far",
        ),
        FeatureGroup(
            columns=(
                FeatureColumn("path_proximity", "proximity"),
                FeatureColumn("tests_per_file", "proximity"),
                FeatureColumn("filename_match", "proximity"),
            ),
            needs=("diff", "test_ids"),
            produce=_proximity_group,
            note="co-location and naming, which assume tests live beside the code",
        ),
        FeatureGroup(
            columns=(FeatureColumn("test_duration", "static_size"),),
            needs=("durations", "test_ids"),
            produce=_duration_group,
            note="wall-clock; hardware-dependent, so available but not comparable",
        ),
        FeatureGroup(
            columns=(
                FeatureColumn("test_lines", "static_size"),
                FeatureColumn("test_tokens", "static_size"),
            ),
            needs=("test_source", "test_ids"),
            produce=_test_size_group,
            note="test size, from the test's own source",
        ),
        FeatureGroup(
            columns=(
                FeatureColumn("code_churn", "static_size"),
                FeatureColumn("added_lines", "static_size"),
                FeatureColumn("removed_lines", "static_size"),
            ),
            needs=("diff",),
            produce=_change_size_group,
            note="change size, parsed from the unified diff",
        ),
        FeatureGroup(
            columns=(
                FeatureColumn("cumulative_failure_rate", "temporal"),
                FeatureColumn("cumulative_runs", "temporal"),
                FeatureColumn("failure_recency", "temporal"),
            ),
            needs=("labels", "execution"),
            produce=_temporal_group,
            note="cumulative over strictly earlier changes; needs a real order to mean anything",
        ),
    ),
)


def structured(
    ds: Dataset,
    *,
    temporal: bool | None = None,
    block: FeatureBlock | None = None,
    diagnostics: Diagnostics | None = None,
) -> FeatureMatrix:
    """Build the structured block against ``ds``.

    ``temporal`` selects the cumulative-failure-history columns. ``None`` means *the default*: on
    for an :attr:`Ordering.NATURAL` dataset, off for an :attr:`Ordering.SYNTHETIC` one,
    where a cumulative feature over an arbitrary order manufactures a leak rather than
    merely failing to be interpretable.

    Study entry points that reproduce the documented numbers opt in explicitly, and the
    accompanying diagnostic travels with the returned matrix rather than being stored on
    the dataset -- so the record describes the derivation, not the call history.
    """
    collected = diagnostics if diagnostics is not None else Diagnostics()
    use_temporal = ds.ordering() is Ordering.NATURAL if temporal is None else bool(temporal)

    if use_temporal and ds.ordering() is Ordering.SYNTHETIC:
        collected.add(
            "feature.temporal_on_synthetic_order",
            Policy.NATURAL_ORDER.value,
            f"temporal features are enabled on a synthetic order (order_seed={ds.order_seed}); "
            "the values are only interpretable under that seed and should be reported as a "
            "spread over seeds, not as one number",
            scope="features:temporal",
        )

    chosen = block if block is not None else STRUCTURED
    if not use_temporal and chosen.has_family("temporal"):
        # Withholding a family the block does not declare is an error, not a no-op, so a
        # custom block without a temporal family is left alone rather than rejected here.
        chosen = chosen.without_families("temporal")
    return chosen.build(ds, collected)


__all__ = ["FAMILIES", "STRUCTURED", "structured"]
