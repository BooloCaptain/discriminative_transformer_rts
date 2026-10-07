"""The kernel subset mechanism plus the example study's subsets, for the kernel tests.

The kernel ships :data:`~rts.data.subsets.DETECTABLE` and the registry mechanism; the study's
deployment proxies live in :mod:`examples.subsets`. Tests resolve names against the composed
registry by default, which is what a study does.
"""

from __future__ import annotations

from examples.subsets import (  # noqa: F401
    COLD_START,
    LOW_COOCCURRENCE,
    NO_PRIOR_FAILURE,
    STUDY,
    cold_start,
    cold_start_mask,
    low_cooccurrence,
    low_cooccurrence_mask,
)
from rts.data.subsets import (  # noqa: F401
    DEFAULT,
    DETECTABLE,
    Subset,
    SubsetRegistry,
)
from rts.data.subsets import resolve as _resolve
from rts.data.subsets import subset as _subset


def subset(name, registry=None):
    """Resolve a subset name against the study registry by default."""
    return _subset(name, registry or STUDY)


def resolve(spec, registry=None):
    """Resolve a subset or name against the study registry by default."""
    return _resolve(spec, registry or STUDY)


__all__ = [
    "COLD_START",
    "DEFAULT",
    "DETECTABLE",
    "LOW_COOCCURRENCE",
    "NO_PRIOR_FAILURE",
    "STUDY",
    "Subset",
    "SubsetRegistry",
    "cold_start",
    "cold_start_mask",
    "low_cooccurrence",
    "low_cooccurrence_mask",
    "resolve",
    "subset",
]
