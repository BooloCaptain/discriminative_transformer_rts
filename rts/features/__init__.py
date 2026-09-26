"""Feature engineering: individually addressable derived features, and named blocks.

The package splits three ways, and the split is the fix for "adding a feature means
editing a central function":

``derived``   one function per quantity, pure over the material it reads.
``block``     the machinery for declaring a block of columns and building a matrix.
``structured``/``bundle``  the two blocks the study actually uses, declared as data.

Adding a column is therefore: add the computation to ``derived``, add a
:class:`~rts.features.block.FeatureColumn` to a group in the block that wants it. No
registry, no import-order coupling, no central assembly function to thread it through.

The two entry points are :func:`structured` and :func:`bundle`, each returning a
:class:`~rts.features.block.FeatureMatrix` -- which carries its own column names, and
records every column it had to coerce, so the coercion is visible where the features are
used rather than discovered later in a table.
"""

from __future__ import annotations

from . import derived, text
from .block import FeatureBlock, FeatureColumn, FeatureGroup, FeatureMatrix
from .bundle import BUNDLE, base_material
from .bundle import bundle as bundle_features
from .structured import FAMILIES, STRUCTURED, structured

#: The two blocks, by name, for callers that dispatch on a string.
BLOCKS: dict[str, FeatureBlock] = {STRUCTURED.name: STRUCTURED, BUNDLE.name: BUNDLE}

__all__ = [
    "BLOCKS",
    "BUNDLE",
    "FAMILIES",
    "STRUCTURED",
    "FeatureBlock",
    "FeatureColumn",
    "FeatureGroup",
    "FeatureMatrix",
    "base_material",
    "bundle_features",
    "derived",
    "structured",
    "text",
]
