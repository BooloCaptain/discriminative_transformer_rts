"""The experiment layer: a declared sweep over five roles, and the run that measures it.

The harness already had three of the five things an experiment sweeps over as plain values
with declarations -- :class:`~rts.data.contract.Dataset`, ``FeatureBlock`` and
:class:`~rts.data.populations.Population` -- and it had no layer that composed them. Instead each
study arm was a hand-written driver that rebuilt the same sequence and kept the study's choices
as its own module constants. The sweep existed as control flow rather than as data, and "a named
variant of one input" was implemented three times (the ladder's rungs, bundle rungs, and the
instruction variants). Two of those drivers are now renderers over a declared arm
(``rts/render/pipeline.py``, ``rts/render/ladder.py``); the rest are listed in ``docs/experiment.md`` §13.

This module is that missing layer. See ``docs/experiment.md`` for the design and its rationale.

**Five roles.** An experiment sweeps ``dataset``, ``features``, ``model``, ``population`` and
``split``. A new *element* in any role is cheap -- that is what most new experimental
dimensions are. A new *role* is a deliberate kernel change, because a role has to say what it
feeds. Every factor the study has conceived already reduces to one of the five: a rung is a
feature block with families withheld, an instruction variant is a model reading a different
cache, a bundle rung is a dataset derived from a base, a starvation threshold is a population,
and a label source is a dataset.

**Cells, not a product.** A cell is one point in the product, and it is either measured or
carries an :class:`~rts.data.contract.Unmeasured` naming what stopped it. Unmeasured cells are
*reported*, never dropped: dropping is what turns "we asked and could not answer" into a
silently halved comparison, which is the failure mode ``docs/refactor.md`` §6 exists to prevent.

**Two authorities.** The *element* declares its name, cost tier and estimated seconds -- facts
about this run's use of a thing. The *value* declares its requirements (``Selector.requirements()``,
``FeatureGroup.needs``, ``Population.needs``, ``Dataset.capabilities()``) -- facts about the
thing itself, written where the material is read so they cannot drift from it.

**Elements are materialised once per run** and shared across the cells that use them, which is
what makes a 48-cell grid affordable: the dataset is built once. This is sound because these
are values (``docs/refactor.md`` §2). The one consequence is that a stateful selector's post-hoc
attributes describe only its most recent call, so importances are captured at the point of
scoring rather than read back afterwards.
"""

__all__ = [
    "ARTIFACT_PREFIX",
    "ROLES",
    "ROLE_DATASET",
    "ROLE_FEATURES",
    "ROLE_MODEL",
    "ROLE_POPULATION",
    "ROLE_SPLIT",
    "ROW_CHANGING_ROLES",
    "Axis",
    "Binding",
    "Cell",
    "CellResult",
    "Comparison",
    "Element",
    "Environment",
    "Experiment",
    "Knobs",
    "RunReport",
    "constant",
    "run",
    "unavailable",
]

from .declaration import (
    ARTIFACT_PREFIX,
    ROLE_DATASET,
    ROLE_FEATURES,
    ROLE_MODEL,
    ROLE_POPULATION,
    ROLE_SPLIT,
    ROLES,
    ROW_CHANGING_ROLES,
    Axis,
    Binding,
    Cell,
    Comparison,
    Element,
    Environment,
    Experiment,
    Knobs,
    constant,
    unavailable,
)
from .report import (
    CellResult,
    RunReport,
)
from .run import (
    run,
)
