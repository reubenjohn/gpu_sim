"""GPU-resident simulation and RL engine. CLI entry point: `uv run gpu-sim <subcommand>`."""

import sys

from gpu_sim import bench

COMMANDS = {"bench": bench.main}


def main() -> int:
    name, *rest = sys.argv[1:] or ["--help"]
    if name not in COMMANDS:
        print(f"usage: gpu-sim {{{','.join(COMMANDS)}}} ...", file=sys.stderr)
        return 2
    return COMMANDS[name](rest)
