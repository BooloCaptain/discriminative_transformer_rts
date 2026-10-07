"""The bundle block: features of a *set* of base changes, presented as one change.

This used to be a second, undeclared feature generator living inside
``rts/bundles.py`` -- thirteen names, a bespoke assembly loop, and four separate
``{n: i for i, n in enumerate(...)}`` reconstructions. It was invisible to the
structured block, so adding a structured feature could not reach it, and renaming one
broke it at runtime rather than at import.

It is a :class:`FeatureBlock` now, declared the same way the structured block is, and it
reaches the base's columns through :class:`FeatureMatrix` rather than by position. What
it needs from its caller -- the bundles themselves, and the base's built features -- is
declared as external inputs, so the block states its inputs instead of assuming them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from rts.features import structured
from rts.features.block import FeatureBlock, FeatureColumn, FeatureGroup

#: Base columns the bundle block reads, by name. Declared, so a rename fails loudly.
BASE_COLUMNS = (
    "function_coverage",
    "filename_match",
    "path_proximity",
    "code_churn",
    "test_duration",
    "test_lines",
    "test_tokens",
    "tests_per_file",
    "cumulative_failure_rate",
    "cumulative_runs",
    "failure_recency",
)


def base_inputs(bundle_ds, *, temporal: bool = True) -> dict[str, Any]:
    """The external inputs this block needs, resolved for ``bundle_ds``.

    Building the base's structured block here rather than inside the group keeps the
    nesting explicit: a bundle feature is a function of the base's features, and the
    base's block is built once for the whole bundle block instead of once per group.
    """
    return {
        "bundles": bundle_ds.changes,
        "base": bundle_ds.base,
        "base_features": structured.structured(bundle_ds.base, temporal=temporal),
    }


def _columns(inputs: Mapping[str, Any]) -> Any:
    """The base's feature matrix, with every declared column touched once.

    Resolving the names here means a rename in the structured block fails at the point
    the bundle block is built, naming the column and the block -- rather than silently
    reading a neighbouring column, which is what the old positional lookups did.
    """
    matrix = inputs["base_features"]
    for name in BASE_COLUMNS:
        matrix.index(name)
    return matrix


def _shape_group(inputs: Mapping[str, Any]) -> Sequence[np.ndarray]:
    """How much of the bundle is observable, and how close it sits to each test."""
    bundles = inputs["bundles"]
    base = _columns(inputs)
    coverage = base.column("function_coverage")
    name_match = base.column("filename_match")
    distance = base.column("path_proximity")
    size_per_change = base.column("code_churn")
    n_b, n_t = len(bundles), coverage.shape[1]

    n_mutations = np.zeros((n_b, n_t), dtype=np.float32)
    n_covered = np.zeros((n_b, n_t), dtype=np.float32)
    frac_covered = np.zeros((n_b, n_t), dtype=np.float32)
    name_any = np.zeros((n_b, n_t), dtype=np.float32)
    min_distance = np.zeros((n_b, n_t), dtype=np.float32)
    code_churn = np.zeros((n_b, n_t), dtype=np.float32)

    for b, bundle in enumerate(bundles):
        members = np.array(bundle.members, dtype=np.int64)
        count = coverage[members].sum(axis=0).astype(np.float32)
        n_mutations[b] = float(len(members))
        n_covered[b] = count
        frac_covered[b] = count / max(len(members), 1)
        name_any[b] = name_match[members].any(axis=0).astype(np.float32)
        min_distance[b] = distance[members].min(axis=0)
        # The concatenated diff preserves line structure, so the bundle's size is the sum
        # over its members -- read from the same base column a single change reports,
        # rather than recomputed by a parallel path that could drift from it.
        code_churn[b] = size_per_change[members].sum(axis=0)

    return n_mutations, n_covered, frac_covered, name_any, min_distance, code_churn


def _test_group(inputs: Mapping[str, Any]) -> Sequence[np.ndarray]:
    """Static-size columns, read from the base at the focal change."""
    bundles = inputs["bundles"]
    base = _columns(inputs)
    names = (
        "test_duration",
        "test_lines",
        "test_tokens",
        "tests_per_file",
    )
    out = []
    for name in names:
        column = base.column(name)
        out.append(
            np.array([column[bundle.focal] for bundle in bundles], dtype=np.float32)
        )
    return out


def _temporal_group(inputs: Mapping[str, Any]) -> Sequence[np.ndarray]:
    """Temporal columns, read from the base at the focal change."""
    bundles = inputs["bundles"]
    base = _columns(inputs)
    out = []
    for name in ("cumulative_failure_rate", "cumulative_runs", "failure_recency"):
        column = base.column(name)
        out.append(
            np.array([column[bundle.focal] for bundle in bundles], dtype=np.float32)
        )
    return out


BUNDLE = FeatureBlock(
    name="bundle",
    note="features of a set of base changes, with the focal change's label held fixed",
    groups=(
        FeatureGroup(
            columns=(
                FeatureColumn("n_mutations", "bundle_shape"),
                FeatureColumn("n_mutations_covered", "bundle_shape"),
                FeatureColumn("frac_mutations_covered", "bundle_shape"),
                FeatureColumn("filename_match_any", "bundle_shape"),
                FeatureColumn("min_path_proximity", "bundle_shape"),
                FeatureColumn("code_churn", "bundle_shape"),
            ),
            needs=("bundles", "base", "base_features"),
            produce=_shape_group,
            note="size and observability of the bundled edit",
        ),
        FeatureGroup(
            columns=(
                FeatureColumn("test_duration", "static_size"),
                FeatureColumn("test_lines", "static_size"),
                FeatureColumn("test_tokens", "static_size"),
                FeatureColumn("tests_per_file", "static_size"),
            ),
            needs=("bundles", "base_features"),
            produce=_test_group,
            note="static size, taken at the focal change",
        ),
        FeatureGroup(
            columns=(
                FeatureColumn("cumulative_failure_rate", "temporal"),
                FeatureColumn("cumulative_runs", "temporal"),
                FeatureColumn("failure_recency", "temporal"),
            ),
            needs=("bundles", "base_features"),
            produce=_temporal_group,
            note="temporal, taken at the focal change",
        ),
    ),
)


def bundle(ds, *, temporal: bool = True, diagnostics=None):
    """Build the bundle block for ``ds``, supplying the base inputs it declares."""
    return BUNDLE.build(
        ds, diagnostics=diagnostics, extra=base_inputs(ds, temporal=temporal)
    )


__all__ = ["BASE_COLUMNS", "BUNDLE", "base_inputs", "bundle"]
