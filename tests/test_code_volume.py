import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "code_volume", Path(__file__).parent.parent / "scripts" / "code_volume.py"
)
code_volume = importlib.util.module_from_spec(spec)
spec.loader.exec_module(code_volume)

ADVICE = (
    "Code volume approximates complexity; do not game it (golfing, packing lines, dropping names). "
    "Simplify, reuse a library, or ask the owner to raise the budget at session start."
)


@pytest.mark.parametrize(
    ("files", "budget"),
    [(code_volume.core_files, code_volume.CORE_BUDGET), (code_volume.all_files, code_volume.TOTAL_BUDGET)],
    ids=["core", "total"],
)
def test_code_volume_within_budget(files, budget):
    sizes = code_volume.sizes(files())
    current = sum(sizes.values())
    top = "\n".join(f"  {n:6d}  {name}" for name, n in sorted(sizes.items(), key=lambda item: -item[1])[:5])
    assert current <= budget, f"{current} AST nodes > budget {budget}. Largest files:\n{top}\n{ADVICE}"


def test_free_constructs_cost_nothing():
    plain = "def f(a, b):\n    return a + b\n"
    decorated = (
        '"""Doc."""\nimport os\n\n\ndef f(a: int, b: int) -> int:\n    """Doc."""\n    # comment\n    return a + b\n'
    )
    assert code_volume.nodes(code_volume.ast.parse(plain)) == code_volume.nodes(code_volume.ast.parse(decorated))
