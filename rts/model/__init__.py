"""The model half: what turns a change and a test into a score.

The model half was deliberately deferred -- "rankers, their context object,
feature-family ablation, model inputs" -- on the argument that it can land independently of
the dataset contract. It has since grown to five modules, which is why it is a package now
rather than a loose ``models.py``:

* :mod:`.rankers` -- the ``Ranker`` interface, its :class:`~rts.model.rankers.Context`,
  the baselines, the trees, :class:`~rts.model.rankers.CachedScores` (a score matrix read
  from a file) and :class:`~rts.model.rankers.ProducedScores` (one produced on first use).
  Declaring what a ranker reads is :meth:`~rts.model.rankers.Ranker.requirements`, the
  half of the contract the experiment layer resolves at the design point boundary.
* :mod:`.semif` -- the pinned reranker adapter: pair construction, cost estimation, and the
  score-cache format its loader parses.
* :mod:`.semif_runner` -- the pairwise scorer (and the context-driven entry point the layer
  calls), which is what actually spends GPU time.
* :mod:`.direct_runner` -- the P1 direct-mode scorer, the task formulation SemIf was
  designed for.
* :mod:`.embed` -- the P3 code-embedding baseline.

Everything here reads the dataset contract through :mod:`rts.data` and writes nothing back,
so a ranker can be evaluated against any dataset without either half knowing about the
other.
"""
