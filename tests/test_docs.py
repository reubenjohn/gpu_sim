"""Every relative link in the project's Markdown docs must resolve to a file."""

import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
LINK = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")


def test_relative_links_resolve():
    docs = [*ROOT.glob("*.md"), *ROOT.glob("docs/*.md")]
    broken = []
    for doc in docs:
        for target in LINK.findall(doc.read_text()):
            if "://" not in target and not (doc.parent / target).exists():
                broken.append(f"{doc.relative_to(ROOT)} -> {target}")
    assert not broken
