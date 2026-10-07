"""The dataset half: what a dataset is, where one comes from, and how they compose.

Everything here speaks the contract in :mod:`.contract`, and each module is a different
question rather than a different type:

* :mod:`.contract` -- the primitives, the declarations, and the values they speak in
  (``Undefined``, ``Diagnostics``, ``Requirement``). Nothing computed from them.
* :mod:`.accessors` -- the derived quantities, as harness functions over the contract, plus
  the one :data:`~rts.data.accessors.INPUTS` catalogue that a group's or a subset's
  ``needs`` resolves through.
* :mod:`.splits` -- the train/test split, which is evaluation configuration rather than a
  property of the data, and the window guard that ties a metric to it.
* :mod:`.subsets` -- named subsets of the evaluation window, and the registry mechanism.
* :mod:`.composition` -- namespacing, pooling and derived datasets, all iteration.

Concrete datasets and their sources are plugins, not part of this package: a study supplies them
(see ``examples/``). The dataset-description and audit helpers live in :mod:`rts.reporting`.
"""
