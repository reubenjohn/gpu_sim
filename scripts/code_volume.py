"""Code-volume budget: count AST nodes in project Python against two budgets.

Nodes are what `ast.walk` yields, minus expression contexts (Load/Store/Del), imports, bare string
statements (docstrings) and type annotations (parameter, return and `x: T`, with everything inside),
so names, comments, docstrings, type hints and line breaks cost nothing.
Gated by tests/test_code_volume.py; `uv run python scripts/code_volume.py` prints the table.
"""

import ast
import sys
from pathlib import Path

# Set by the owner (2026-11-14); raised only at phase gates, by the owner.
CORE_BUDGET = 2500  # src/gpu_sim/**/*.py
TOTAL_BUDGET = 6000  # all project Python: src/, scripts/, tests/, any other non-hidden dir
ROOT = Path(__file__).resolve().parent.parent
ANNOTATIONS = {ast.arg: "annotation", ast.AnnAssign: "annotation"}
ANNOTATIONS |= {ast.FunctionDef: "returns", ast.AsyncFunctionDef: "returns"}


def skipped(node: ast.AST) -> bool:
    docstring = isinstance(node, ast.Expr) and isinstance(getattr(node.value, "value", None), str)
    return docstring or isinstance(node, ast.Load | ast.Store | ast.Del | ast.Import | ast.ImportFrom)


def nodes(tree: ast.AST) -> int:
    if skipped(tree):
        return 0
    skip = ANNOTATIONS.get(type(tree))
    children = [c for field, c in ast.iter_fields(tree) if field != skip]
    flat = [c for value in children for c in (value if isinstance(value, list) else [value])]
    return int(not isinstance(tree, ast.Module)) + sum(nodes(c) for c in flat if isinstance(c, ast.AST))


def count(path: Path) -> int:
    return nodes(ast.parse(path.read_text(), str(path)))


def core_files() -> list[Path]:
    return sorted((ROOT / "src" / "gpu_sim").rglob("*.py"))


def all_files() -> list[Path]:
    """Every .py under ROOT outside hidden dirs (.venv, .git, ...)."""
    return sorted(p for p in ROOT.rglob("*.py") if not any(part.startswith(".") for part in p.relative_to(ROOT).parts))


def sizes(files: list[Path]) -> dict[str, int]:
    return {str(f.relative_to(ROOT)): count(f) for f in files}


def main() -> int:
    total = sizes(all_files())
    for name, n in sorted(total.items(), key=lambda item: -item[1]):
        print(f"{n:6d}  {name}")
    core = sum(sizes(core_files()).values())
    print(f"{core:6d}  core total (budget {CORE_BUDGET})")
    print(f"{sum(total.values()):6d}  project total (budget {TOTAL_BUDGET})")
    return int(core > CORE_BUDGET or sum(total.values()) > TOTAL_BUDGET)


if __name__ == "__main__":
    sys.exit(main())
