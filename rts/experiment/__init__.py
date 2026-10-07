"""The experiment layer: a declared sweep over roles, and the run that measures it.

An *experiment* is a value: a named product of levels over its roles, with a study's choices
expressed as data rather than as control flow. This package is the mechanism that makes that
possible -- :mod:`.declaration` says what an experiment is, :mod:`.run` measures it, and
:mod:`.report` records what was measured. A study declares one; a plugin package supplies the
datasets, features, rankers and subsets it names.

**Roles.** An experiment sweeps ``dataset``, ``features``, ``model``, ``subset``, ``split`` and
``budget``. A new *level* in any role is cheap -- that is what most new experimental dimensions
are. A new *role* is a deliberate kernel change, because a role has to say what it feeds. Every
factor a study has conceived reduces to one of the roles: a rung is a feature block with families
withheld, an instruction variant is a model reading a different cache, a bundle rung is a dataset
derived from a base, a starvation threshold is a subset, a label source is a dataset, and a
budget is an evaluation threshold.

**Design points, not a product.** A design point is one point in the product, and it is either
measured or carries an :class:`~rts.data.contract.Undefined` naming what stopped it. Undefined
design points are *reported*, never dropped: dropping is what turns "we asked and could not
answer" into a silently halved contrast.

**Two authorities.** The *level* declares its name, cost tier and estimated seconds -- facts about
this run's use of a thing. The *value* declares its requirements (``Ranker.requirements()``,
``FeatureGroup.needs``, ``Subset.needs``, ``Dataset.capabilities()``) -- facts about the thing
itself, written where the input is read so they cannot drift from it.

**Elements are materialised once per run** and shared across the design points that use them,
which is what makes a large grid affordable: the dataset is built once. This is sound because
these are values. The one consequence is that a stateful ranker's post-hoc attributes describe
only its most recent call, so importances are captured at the point of scoring rather than read
back afterwards.
"""

__all__ = [
    "ARTIFACT_PREFIX",
    "FACTORS",
    "FACTOR_DATASET",
    "FACTOR_FEATURES",
    "FACTOR_MODEL",
    "FACTOR_SUBSET",
    "FACTOR_SPLIT",
    "FACTOR_BUDGET",
    "ROW_CHANGING_FACTORS",
    "Factor",
    "Binding",
    "Builder",
    "DesignPoint",
    "DesignPointResult",
    "Contrast",
    "Level",
    "Environment",
    "Experiment",
    "Controls",
    "RunReport",
    "budget_level",
    "constant",
    "declared_budgets",
    "run",
    "split_level",
    "unavailable",
]

from .declaration import (
    ARTIFACT_PREFIX,
    FACTOR_BUDGET,
    FACTOR_DATASET,
    FACTOR_FEATURES,
    FACTOR_MODEL,
    FACTOR_SPLIT,
    FACTOR_SUBSET,
    FACTORS,
    ROW_CHANGING_FACTORS,
    Binding,
    Builder,
    Contrast,
    Controls,
    DesignPoint,
    Environment,
    Experiment,
    Factor,
    Level,
    budget_level,
    constant,
    declared_budgets,
    split_level,
    unavailable,
)
from .report import (
    DesignPointResult,
    RunReport,
)
from .run import (
    run,
)
