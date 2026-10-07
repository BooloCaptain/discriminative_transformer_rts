"""``python -m rts.studies`` -- run a declared condition.

A package needs an explicit ``__main__`` for ``-m`` to work; the module version used
to carry the ``if __name__ == "__main__"`` block directly.
"""

from . import main

if __name__ == "__main__":
    main()
