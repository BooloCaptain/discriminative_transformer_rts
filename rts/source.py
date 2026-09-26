"""Extract test-function source text by pytest node id.

Node ids look like ``tests/test_validate.py::test_email_repr`` or
``tests/test_context.py::TestContext::test_context_load_dump``. We parse each
test file once with ``ast`` and slice the function body out of the file.

The extracted text is the "test side" of the unstructured features (BM25 and the
SemIf reranker). Granularity is the test *function*, not the whole file, since
that is what a reranker can score meaningfully and what the plan specifies.

The checkout root is a parameter, not a module-level pin: a dataset supplies it, so
two datasets over two checkouts can coexist (§7 of the refactor design). The
``ast`` parses are cached per ``(checkout, file)``, which is a pure function of the
two, not dataset state.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from . import config


@dataclass(frozen=True)
class TestInfo:
    nodeid: str
    file: str
    qualified_name: str
    source: str
    n_lines: int
    n_tokens: int


def _strip_params(nodeid: str) -> str:
    """Drop a pytest parametrization suffix, e.g. ``test_x[a-1]`` -> ``test_x``."""
    head, _, tail = nodeid.partition("::")
    return head + "::" + tail.split("[", 1)[0]


def _iter_functions(tree: ast.AST):
    """Yield (dotted qualified name, node) for module-level, class, and nested-class functions."""

    def walk(body, prefix: str):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield prefix + node.name, node
            elif isinstance(node, ast.ClassDef):
                yield from walk(node.body, prefix + node.name + ".")

    yield from walk(tree.body, "")


@lru_cache(maxsize=None)
def _functions_in(sut: Path, file_rel: str) -> dict[str, ast.AST]:
    path = sut / file_rel
    if not path.exists():
        return {}
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return {}
    return dict(_iter_functions(tree))


def _source_of(sut: Path, file_rel: str, node: ast.AST) -> str:
    lines = (sut / file_rel).read_text().splitlines()
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    end = node.end_lineno or node.lineno
    return "\n".join(lines[start - 1 : end])


@lru_cache(maxsize=None)
def _test_info(nodeid: str, sut: Path) -> TestInfo | None:
    stripped = _strip_params(nodeid)
    file_rel, _, name = stripped.partition("::")
    # Node ids nest with ``::``; the ast keys nest with ``.``.
    name = name.replace("::", ".")
    if not name:
        return None
    funcs = _functions_in(sut, file_rel)
    node = funcs.get(name)
    if node is None:
        return None
    source = _source_of(sut, file_rel, node)
    return TestInfo(
        nodeid=nodeid,
        file=file_rel,
        qualified_name=name,
        source=source,
        n_lines=source.count("\n") + 1,
        n_tokens=len(source.split()),
    )


def _root(sut: Path | None) -> Path:
    return Path(sut) if sut is not None else config.SUT


def test_info(nodeid: str, sut: Path | None = None) -> TestInfo | None:
    return _test_info(nodeid, _root(sut))


def test_source(nodeid: str, sut: Path | None = None) -> str | None:
    """Source text of one test function, or ``None`` when it cannot be located.

    ``None`` and ``""`` are different states and the harness keeps them apart: an
    unlocatable test contributes no size features, an empty one contributes
    zero-length ones.
    """
    info = _test_info(nodeid, _root(sut))
    return info.source if info is not None else None


def load_all(nodeids: list[str], sut: Path | None = None) -> dict[str, TestInfo]:
    """Map node id -> TestInfo, skipping ids whose source cannot be located."""
    root = _root(sut)
    out: dict[str, TestInfo] = {}
    for nodeid in nodeids:
        info = _test_info(nodeid, root)
        if info is not None:
            out[nodeid] = info
    return out


def exists(file_rel: str, sut: Path | None = None) -> bool:
    return (_root(sut) / file_rel).exists()


if __name__ == "__main__":
    from . import datasets

    ds = datasets.marshmallow()
    infos = load_all(ds.test_ids)
    print(f"node ids      : {len(ds.test_ids)}")
    print(f"resolved      : {len(infos)}")
    print(f"unresolved    : {len(ds.test_ids) - len(infos)}")
    for missing in [i for i in ds.test_ids if i not in infos][:5]:
        print("  missing:", missing)
    if infos:
        sample = infos[sorted(infos)[0]]
        print("\n--- sample ---")
        print("nodeid :", sample.nodeid)
        print("name   :", sample.qualified_name)
        print("lines  :", sample.n_lines, "| tokens:", sample.n_tokens)
        print(sample.source)
