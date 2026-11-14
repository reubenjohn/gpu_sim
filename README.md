# gpu_sim

A massively parallel, 100% GPU-resident simulation and RL engine for one NVIDIA GTX 1070
(Pascal SM 6.1, 8 GB, WSL2). It starts with a hummingbird/flower co-evolution sanity check and
scales into multi-agent town traffic RL. The aim is millions of generations or env steps per
hour, with no host↔device traffic in the loop.

## Quick start

```sh
uv sync
uv run pytest -q
uv run gpu-sim bench -- <workload command>   # needs the GPU, gcc and CUPTI
```

The `bench` workload contract is in the docstring of `src/gpu_sim/bench.py`.

## Requirements

- An NVIDIA driver with WSL2 GPU support (`nvidia-smi` works)
- CUDA 12.x toolkit with CUPTI (default `/usr/local/cuda`, override with `CUDA_HOME`) and `gcc`,
  for `gpu-sim bench` (it builds its CUPTI counter on first use into `~/.cache/gpu_sim/`)
- [uv](https://docs.astral.sh/uv/)

## Docs

- [docs/BRIEF.md](docs/BRIEF.md): goals, constraints, targets, autonomy boundary (owner-maintained)
- [docs/BACKLOG.md](docs/BACKLOG.md): prioritised work, next gate first
- [DECISIONS.md](DECISIONS.md): current technical decisions, evidence, sweeps and rejected options
