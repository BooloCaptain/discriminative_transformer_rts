"""The dataset half: what a dataset is, where one comes from, and how they compose.

Everything here speaks the contract in :mod:`.contract`, and the division is the one
``docs/refactor.md`` argues for -- each module is a different question rather than a
different type:

* :mod:`.contract` -- the primitives, the declarations, and the values they speak in
  (``Unmeasured``, ``Warnings``, ``Requirement``). Nothing computed from them.
* :mod:`.accessors` -- the derived quantities, as harness functions over the contract, plus
  the one :data:`~rts.data.accessors.MATERIAL` catalogue that a group's or a population's
  ``needs`` resolves through.
* :mod:`.splits` -- the train/test split, which is evaluation configuration rather than a
  property of the data, and the window guard that ties a metric to it.
* :mod:`.populations` -- named subsets of the evaluation window.
* :mod:`.composition` -- namespacing, pooling and derived datasets, all iteration.
* :mod:`.datasets` -- the concrete datasets: three pieces of machinery behind the contract.
* :mod:`.sources` -- the raw material for one SUT or revision, holding no module-level state.
* :mod:`.mutmut` -- mutmut's raw artifacts and the sample generator that turns them into
  changes (the ``mutmut`` and ``full`` label sources).
* :mod:`.test_source` -- test-function source text by pytest node id, cached per checkout.
* :mod:`.reporting` -- describing a dataset, and auditing what a consumer should know.

Two names were one letter apart and are not any more: ``rts.source`` and ``rts.sources`` are
now :mod:`.test_source` and :mod:`.sources`. The module that reads mutmut's files is
:mod:`.mutmut` rather than ``rts.artifacts``, which read like the recorded ``artifacts/``
directory it never touches.
"""
