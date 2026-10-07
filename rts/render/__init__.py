"""Renderers: what runs a declared experiment, and what reads a recorded artifact.

Nothing here is part of the measurement. A renderer *calls* the harness -- it runs a condition
declared in :mod:`rts.studies` and writes the artifact the recorded numbers are written
against, or it reads a recorded artifact and draws a figure. That is why the package exists:
the same five modules used to sit beside the values they consume, so ``rts/render/pipeline.py``
read like part of the harness rather than like the script that writes
``artifacts/results_full.json``.

Two kinds, and the distinction is stated in each module's docstring:

* **drivers** -- :mod:`.pipeline`, :mod:`.ladder`, :mod:`.variations`, :mod:`.bugsinpy`. Each
  contains no experiment logic: it runs an ``Experiment`` value from :mod:`rts.studies` and
  renders the payload its recorded artifact, the tables in ``docs/implementation.md`` and
  the figures are written against. ``scripts/verify_experiment_layer.py`` is the gate that
  compares what they render against what is recorded, leaf by leaf.
* **readers** -- :mod:`.panels`, :mod:`.figures`. These read recorded artifacts or score
  caches and compute nothing that is a design point: a panel bins the *dataset* and evaluates each
  model on the subset of a bin it has scores for (a per-*row* filter, where
  ``Level.applies`` is per-*design point*), and a figure is a reading of recorded numbers.

``bundles`` is the one driver still outside this package, because ``studies/factors.py`` needs
its rung *definitions* and ``datasets.bundles()`` needs its bundle construction -- moving it
here would make the declarations import a renderer, which ``tests/test_studies.py`` forbids.
Its migration (rungs as a dataset factor, the block in ``features/bundle.py``) is what removes
that, and is ``docs/experiment.md`` §13's item 2.
"""
