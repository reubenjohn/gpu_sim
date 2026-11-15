"""Doc invariants JSON Schema and lychee can't express. Links: lychee; backlog shape: check-jsonschema."""

import re
from pathlib import Path

import yaml

import gpu_sim

ROOT = Path(__file__).parents[1]
DOCS = [path for path in ROOT.rglob("*.md") if ".venv" not in path.parts]
BACKLOG = yaml.safe_load((ROOT / "docs/backlog.yaml").read_text())["items"]


def test_backlog_caps_and_references():
    ids = [item["id"] for item in BACKLOG]
    statuses = [item["status"] for item in BACKLOG]
    referenced = {ref for item in BACKLOG for ref in [*item["blocked_by"], *re.findall(r"B\d+", item["serves"])]}
    assert len(ids) == len(set(ids)), "duplicate backlog ids"
    assert statuses.count("active") <= 15 and statuses.count("parked") <= 10, "backlog caps (BACKLOG.md rule 6)"
    assert referenced <= set(ids), f"unknown ids in blocked_by/serves: {sorted(referenced - set(ids))}"


def test_documented_commands_exist():
    text = "\n".join(doc.read_text() for doc in DOCS)
    paths = re.findall(r"uv run python ([\w./-]+\.py)", text)
    subcommands = re.findall(r"uv run gpu-sim ([\w-]+)", text)
    assert [path for path in paths if not (ROOT / path).exists()] == []
    assert set(subcommands) <= set(gpu_sim.COMMANDS)


def test_brief_table_numbers_are_labelled():
    brief = (ROOT / "docs/BRIEF.md").read_text()
    tables = [section for section in brief.split("\n## ") if section.startswith(("Hard constraints", "Targets"))]
    rows = [line for table in tables for line in table.splitlines() if line.startswith("|")]
    body = [row for row in rows if not row.startswith(("|---", "| #", "| Measure"))]
    assert len(body) == 10
    assert [row[:40] for row in body if "(owner" not in row and "(measured" not in row] == []
