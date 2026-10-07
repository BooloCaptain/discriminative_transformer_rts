"""The experiment layer: a declared sweep over five roles, and the run that measures it.

The harness already had three of the five things an experiment sweeps over as plain values
with declarations -- :class:`~rts.data.contract.Dataset`, ``FeatureBlock`` and
:class:`~rts.data.subsets.Subset` -- and it had no layer that composed them. Instead each
study condition was a hand-written driver that rebuilt the same sequence and kept the study's choices
as its own module constants. The sweep existed as control flow rather than as data, and "a named
variant of one input" was implemented three times (the ladder's rungs, bundle rungs, and the
instruction variants). Two of those drivers are now renderers over a declared condition
(``rts/render/pipeline.py``, ``rts/render/ladder.py``); the rest are listed in ``docs/experiment.md`` §13.

This module is that missing layer. See ``docs/experiment.md`` for the design and its rationale.

**Five roles.** An experiment sweeps ``dataset``, ``features``, ``model``, ``subset`` and
``split``. A new *level* in any role is cheap -- that is what most new experimental
dimensions are. A new *role* is a deliberate kernel change, because a role has to say what it
feeds. Every factor the study has conceived already reduces to one of the five: a rung is a
feature block with families withheld, an instruction variant is a model reading a different
cache, a bundle rung is a dataset derived from a base, a starvation threshold is a subset,
and a label source is a dataset.

**Design points, not a product.** A design point is one point in the product, and it is either measured or
carries an :class:`~rts.data.contract.Undefined` naming what stopped it. Undefined design points are
*reported*, never dropped: dropping is what turns "we asked and could not answer" into a
silently halved contrast, which is the failure mode ``docs/refactor.md`` §6 exists to prevent.

**Two authorities.** The *level* declares its name, cost tier and estimated seconds -- facts
about this run's use of a thing. The *value* declares its requirements (``Ranker.requirements()``,
``FeatureGroup.needs``, ``Subset.needs``, ``Dataset.capabilities()``) -- facts about the
thing itself, written where the inputs is read so they cannot drift from it.

**Elements are materialised once per run** and shared across the design points that use them, which is
what makes a 48-design point grid affordable: the dataset is built once. This is sound because these
are values (``docs/refactor.md`` §2). The one consequence is that a stateful ranker's post-hoc
attributes describe only its most recent call, so importances are captured at the point of
scoring rather than read back afterwards.
"""

__all__ = [
    "ARTIFACT_PREFIX",
    "FACTORS",
    "FACTOR_DATASET",
    "FACTOR_FEATURES",
    "FACTOR_MODEL",
    "FACTOR_SUBSET",
    "FACTOR_SPLIT",
    "ROW_CHANGING_FACTORS",
    "Factor",
    "Binding",
    "DesignPoint",
    "DesignPointResult",
    "Contrast",
    "Level",
    "Environment",
    "Experiment",
    "Controls",
    "RunReport",
    "constant",
    "run",
    "unavailable",
]

from .declaration import (
    ARTIFACT_PREFIX,
    FACTOR_DATASET,
    FACTOR_FEATURES,
    FACTOR_MODEL,
    FACTOR_SPLIT,
    FACTOR_SUBSET,
    FACTORS,
    ROW_CHANGING_FACTORS,
    Binding,
    Contrast,
    Controls,
    DesignPoint,
    Environment,
    Experiment,
    Factor,
    Level,
    constant,
    unavailable,
)
from .report import (
    DesignPointResult,
    RunReport,
)
from .run import (
    run,
)
