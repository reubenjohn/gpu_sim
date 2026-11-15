# Decisions

The current technical decisions, with their evidence. Goals, constraints and owner-level
principles are in [docs/BRIEF.md](docs/BRIEF.md); work is in [docs/BACKLOG.md](docs/BACKLOG.md).

Rules:
0. Every decision follows from the BRIEF (a constraint, requirement or target) plus evidence
   judged against its baseline (BRIEF P1 Baseline). Its "why" names which.
1. This file holds only CURRENT decisions: ID | decision | why (one line) | evidence (latest
   numbers + date, or a link to a committed results file). It is not dated log entries.
2. When a decision changes, edit it in place. Delete superseded decisions; git history keeps them.
3. Sweep results: one small table per sweep, overwritten when re-run, plus a one-line takeaway,
   for the current framework(s) only (sweep list: BACKLOG preamble); delete tables for rejected frameworks.
4. Rejected options get a short "Rejected" list, one line each ("X: rejected because Y, date"),
   so agents don't retry them. That is the only history kept visible. Cap: 10 lines; merge or
   drop the oldest, least relevant lines to stay under it.
5. Known gaps or debt belong in the backlog (docs/backlog.yaml), not here.
6. **Owner-gated** decisions (marked below: the hardware and toolchain pins) change only with the
   owner, because a wrong change can break the machine or the toolchain. Agents may edit every
   other decision when they have the evidence.
7. Current measurements: overwrite each row in place with the latest value.

Before proposing a framework or layout change, record its `gpu-sim bench` numbers (median of 3)
as the evidence for that decision here (BRIEF P1). Evidence format:
`steps_per_hour | speedup vs CPU reference | host_memcpys | host_syncs | launches_per_10k | peak_vram_bytes`,
plus commit and workload. Each claim names its baseline (BRIEF P1; per-stage baselines in T13).

## Settled technical decisions

| ID | Decision | Why | Evidence |
|---|---|---|---|
| T1 | The bench gate injects a CUPTI driver-API callback counter (`src/gpu_sim/cucount.c`, via `CUDA_INJECTION64_PATH`) that counts driver calls by name between cuProfilerStart and cuProfilerStop and writes `{name: count}` JSON at exit. `bench.py` builds it with gcc into `~/.cache/gpu_sim/` and classifies names into launches, host memcpys and host syncs. | Exact counts for raw CUDA, JAX and PyTorch where nsys gives none. | 2026-11-14, inside the capture range: raw CUDA `tests/fixtures/probe.cu` 1000 launches, 3 H2D, 2 D2H, 4 syncs (all exact; now the `@pytest.mark.gpu` end-to-end test); JAX 101 `cuLaunchKernel`, 1 H2D, 1 D2H; torch 2.7.1+cu126 200 launches, 1 H2D, 1 D2H, 2 `cuStreamSynchronize` + 1 `cuCtxSynchronize`. |
| T2 | Generic `cuMemcpy`/`cuMemcpyAsync` count as host copies; a run with zero launches fails the gate. | Direction is unknown under unified addressing: a false alarm beats a missed round-trip. Zero launches means the capture range was missed. | `tests/test_bench.py` |
| T3 | Peak VRAM is read from NVML (whole process: CUDA context and framework pools), not an allocator counter. JAX workloads set `XLA_PYTHON_CLIENT_PREALLOCATE=false`. | Allocator counters miss the context and pools, and JAX preallocation would fake a 75% peak. | contract in `src/gpu_sim/bench.py` |
| T4 | **Owner-gated.** Python 3.12 managed by uv (`uv run`, `uv add`); no pip, no system Python. | One reproducible toolchain for agents and CI. | `pyproject.toml`, `uv.lock` |
| T5 | **Owner-gated.** CUDA 12.x only: GPU frameworks must ship CUDA 12.x builds with SM 6.1 kernels; check before adopting any wheel. | CUDA 13 dropped Pascal. | see Rejected |
| T6 | **Owner-gated.** PyTorch is pinned to `torch==2.7.1` from the cu126 index. | The latest wheel lacks sm_6x kernels ("no kernel image"). | 2026-11-14 probe (T1) |
| T7 | **Owner-gated.** No driver past R580: keep the Windows driver on the R580 branch (installed: 581.29). | R580 is the last branch supporting Pascal. | `nvidia-smi` |
| T8 | GitHub CI is CPU-only (ruff, pytest, code volume); `@pytest.mark.gpu` tests skip without the GPU, nvcc or gcc. | No GPU runners at $0 (BRIEF C4, Money budget). | `.github/workflows/ci.yml` |
| T9 | C6 recompiles are counted with `jax.log_compiles` or the `torch._dynamo` recompile counter inside the timed loop. | The frameworks already expose them. | none yet |
| T10 | **The framework is NOT settled.** JAX (`jit` + `vmap` + `lax.scan`), PyTorch (`torch.compile` + `torch.cuda.CUDAGraph`, optionally TorchRL) and custom kernels (Warp/Taichi/Triton/CUDA C++) are hypotheses; the Phase 2 shootout picks the fastest candidate that matches the CPU reference and beats the no-selection control, with bench evidence (BRIEF P1). Hybrids (custom env kernels + a JAX/PyTorch policy) are valid. A candidate with no working SM 6.1 build is dropped and added to Rejected; no owner needed. | Measure, don't assume. | none yet (BACKLOG shootout items) |
| T11 | **Owner-gated.** Target hardware: NVIDIA GTX 1070 (Pascal SM 6.1, 8 GB, 1920 CUDA cores, ~256 GB/s) on WSL2, $0 cloud. | BRIEF C4: owner hardware and cost. | `nvidia-smi --query-gpu=name,compute_cap --format=csv` |
| T12 | FP32 and integers only (BRIEF C5). | FP16 math runs 64× slower than FP32 on this card. | Nvidia's GP104 spec, not measured here; `tests/test_constraints.py` |
| T13 | Per-stage P1 baselines. Correct and Fast: a NumPy CPU reference per stage (Stage 1: B12). Learns: Stage 1 no-selection control (B13); Stage 2 a random policy, then a fixed-timer traffic light. Instrument: the known-answer `tests/fixtures/probe.cu` (T1), and the same workload with and without the counter (B6). | BRIEF P1. | T1; BACKLOG B6, B12, B13 |

## Current measurements

| Measure | Value | Date | Command |
|---|---|---|---|
| Probe counts (raw CUDA `tests/fixtures/probe.cu`) | 1000 launches, 3 H2D, 2 D2H, 4 syncs | 2026-11-14 | `uv run pytest -q -m gpu` |
| Stage 1 CPU reference generations/hour (baseline) | none yet (B12) | | B12's pytest / script |
| Stage 1 generations/hour | none yet | | `uv run gpu-sim bench -- <stage-1 workload>` |

## Open technical questions

Decided by evidence; every answer must avoid dynamic allocation.

- Framework (T10). JAX: env + policy + update can fuse into one XLA loop (PureJaxRL-style); risk:
  XLA behaviour and shared-memory spills on Pascal as traffic branching grows. PyTorch CUDA
  Graphs: near-zero replay dispatch cost on WSL2; risk: graph breaks on resets and neighbour
  lookups. Custom kernels: spatial hashing, raycasts and graph kinematics map naturally to SIMT;
  risk: autodiff and NN integration.
- Topology: a 2D grid with spatial hashing vs a lane graph as fixed adjacency tensors.
- Traffic state layout: fixed tensors such as `[envs, max_vehicles, state]` and
  `[envs, max_intersections, light_state]`; IDM vs bicycle kinematics; intersection conflict
  resolution.
- Observation: kinematic feature vectors vs local occupancy grids (which must not need CNN
  rendering in VRAM).
- Learner: neuroevolution (GA/CMA-ES) vs PPO on the GPU.

## Sweep results

One table per sweep (overwritten on re-run) plus a one-line takeaway. None recorded yet.

## Rejected

- nsys (`--capture-range=cudaProfilerApi` + `nsys stats` CSV reports) as the bench gate: rejected because nsys records no CUDA data at all on WSL2 + GTX 1070, so every report was empty, 2026-11-14.
- Hand-written SQL over the nsys `--export=sqlite` DB: rejected because `nsys stats` already computes the totals and SQL adds code volume, 2026-11-14.
- The latest PyTorch wheel: rejected because it lacks sm_6x kernels ("no kernel image"), 2026-11-14.
- CUDA 13 framework builds: rejected because CUDA 13 dropped Pascal, 2026-11-14.
- Windows drivers newer than the R580 branch: rejected because they drop Pascal, 2026-11-14.
