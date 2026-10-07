"""Concrete plugins for one study, plus a declarative example.

Nothing here is part of the harness: a plugin in this package *implements* one of the
kernel's interfaces (a :class:`rts.data.contract.Dataset`, a
:class:`rts.model.rankers.Ranker`, a
:class:`rts.features.block.FeatureBlock`), and ``example`` combines several of them into a
sweep with no logic of its own.

The dependency runs one way: this package imports ``rts``, and ``rts`` never imports this
package. That is what keeps the harness generic (goal 11) while a study still has real
models and datasets to run.
"""
