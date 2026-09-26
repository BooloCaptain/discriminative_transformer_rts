"""The bundle block: features of a *set* of base changes, presented as one change.

This used to be a second, undeclared feature generator living inside
``rts/bundles.py`` -- thirteen names, a bespoke assembly loop, and four separate
``{n: i for i, n in enumerate(...)}`` reconstructions. It was invisible to the
structured block, so adding a structured feature could not reach it, and renaming one
broke it at runtime rather than at import.

It is a :class:`FeatureBlock` now, declared the same way the structured block is, and it
reaches the base's columns through :class:`FeatureMatrix` rather than by position. What
it needs from its caller -- the bundles themselves, and the base's built features -- is
declared as external material, so the block states its inputs instead of assuming them.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from . import structured
from .block import FeatureBlock, FeatureColumn, FeatureGroup

#: Base columns the bundle block reads, by name. Declared, so a rename fails loudly.
BASE_COLUMNS = (
    "covers_function",
    "filename_stem_match",
    "path_distance",
    "change_size",
    "test_duration",
    "test_n_lines",
    "test_n_tokens",
    "n_tests_in_test_file",
    "test_failure_rate_cum",
    "test_runs_cum",
    "test_last_failure_age",
)


def base_material(bundle_ds, *, history: bool = True) -> dict[str, Any]:
    """The external material this block needs, resolved for ``bundle_ds``.

    Building the base's structured block here rather than inside the group keeps the
    nesting explicit: a bundle feature is a function of the base's features, and the
    base's block is built once for the whole bundle block instead of once per group.
    """
    return {
        "bundles": bundle_ds.changes,
        "base": bundle_ds.base,
        "base_features": structured.structured(bundle_ds.base, history=history),
    }


def _columns(material: Mapping[str, Any]) -> Any:
    """The base's feature matrix, with every declared column touched once.

    Resolving the names here means a rename in the structured block fails at the point
    the bundle block is built, naming the column and the block -- rather than silently
    reading a neighbouring column, which is what the old positional lookups did.
    """
    matrix = material["base_features"]
    for name in BASE_COLUMNS:
        matrix.index(name)
    return matrix


def _shape_group(material: Mapping[str, Any]) -> Sequence[np.ndarray]:
    """How much of the bundle is observable, and how close it sits to each test."""
    bundles = material["bundles"]
    base = _columns(material)
    coverage = base.column("covers_function")
    name_match = base.column("filename_stem_match")
    distance = base.column("path_distance")
    size_per_change = base.column("change_size")
    n_b, n_t = len(bundles), coverage.shape[1]

    n_mutations = np.zeros((n_b, n_t), dtype=np.float32)
    n_covered = np.zeros((n_b, n_t), dtype=np.float32)
    frac_covered = np.zeros((n_b, n_t), dtype=np.float32)
    name_any = np.zeros((n_b, n_t), dtype=np.float32)
    min_distance = np.zeros((n_b, n_t), dtype=np.float32)
    change_size = np.zeros((n_b, n_t), dtype=np.float32)

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
        change_size[b] = size_per_change[members].sum(axis=0)

    return n_mutations, n_covered, frac_covered, name_any, min_distance, change_size


def _test_group(material: Mapping[str, Any]) -> Sequence[np.ndarray]:
    """Test-intrinsic columns, read from the base at the signal change."""
    bundles = material["bundles"]
    base = _columns(material)
    names = (
        "test_duration",
        "test_n_lines",
        "test_n_tokens",
        "n_tests_in_test_file",
    )
    out = []
    for name in names:
        column = base.column(name)
        out.append(
            np.array([column[bundle.signal] for bundle in bundles], dtype=np.float32)
        )
    return out


def _history_group(material: Mapping[str, Any]) -> Sequence[np.ndarray]:
    """History columns, read from the base at the signal change."""
    bundles = material["bundles"]
    base = _columns(material)
    out = []
    for name in ("test_failure_rate_cum", "test_runs_cum", "test_last_failure_age"):
        column = base.column(name)
        out.append(
            np.array([column[bundle.signal] for bundle in bundles], dtype=np.float32)
        )
    return out


BUNDLE = FeatureBlock(
    name="bundle",
    note="features of a set of base changes, with the signal's label held fixed",
    groups=(
        FeatureGroup(
            columns=(
                FeatureColumn("n_mutations", "bundle_shape"),
                FeatureColumn("n_mutations_covered", "bundle_shape"),
                FeatureColumn("frac_mutations_covered", "bundle_shape"),
                FeatureColumn("name_match_any", "bundle_shape"),
                FeatureColumn("min_path_distance", "bundle_shape"),
                FeatureColumn("change_size", "bundle_shape"),
            ),
            needs=("bundles", "base", "base_features"),
            produce=_shape_group,
            note="size and observability of the bundled edit",
        ),
        FeatureGroup(
            columns=(
                FeatureColumn("test_duration", "intrinsic"),
                FeatureColumn("test_n_lines", "intrinsic"),
                FeatureColumn("test_n_tokens", "intrinsic"),
                FeatureColumn("n_tests_in_test_file", "intrinsic"),
            ),
            needs=("bundles", "base_features"),
            produce=_test_group,
            note="test-intrinsic, taken at the signal change",
        ),
        FeatureGroup(
            columns=(
                FeatureColumn("test_failure_rate_cum", "history"),
                FeatureColumn("test_runs_cum", "history"),
                FeatureColumn("test_last_failure_age", "history"),
            ),
            needs=("bundles", "base_features"),
            produce=_history_group,
            note="history, taken at the signal change",
        ),
    ),
)


def bundle(ds, *, history: bool = True, warnings=None):
    """Build the bundle block for ``ds``, supplying the base material it declares."""
    return BUNDLE.build(
        ds, warnings=warnings, extra=base_material(ds, history=history)
    )


__all__ = ["BASE_COLUMNS", "BUNDLE", "base_material", "bundle"]
