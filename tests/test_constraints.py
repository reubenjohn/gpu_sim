"""Code-checkable hard constraints from docs/BRIEF.md."""

import re
from pathlib import Path

SRC = Path(__file__).parents[1] / "src"
HALF_PRECISION = re.compile(r"float16|bfloat16|autocast", re.IGNORECASE)


def test_c5_no_half_precision_in_src():
    """C5: FP32 and integers only, so no fp16/bf16 or autocast anywhere in src/."""
    hits = [
        f"{path.relative_to(SRC)}:{number}"
        for path in SRC.rglob("*")
        if path.is_file() and path.suffix in {".py", ".c", ".cu", ".h"}
        for number, line in enumerate(path.read_text().splitlines(), start=1)
        if HALF_PRECISION.search(line)
    ]
    assert not hits
