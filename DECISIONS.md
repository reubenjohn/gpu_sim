# Decisions and benchmark log

Append-only. Every framework or memory-layout experiment gets an entry with its exact
`uv run gpu-sim bench` output (BRIEF C7) **before** an architectural change is proposed.
Settled decisions are summarised in [docs/BRIEF.md](docs/BRIEF.md) §6.

Entry template:

```
## YYYY-MM-DD: <experiment or decision>
Commit: <sha>  Workload: <command>  Runs: 3 (median reported)
steps_per_hour | host_memcpys | launches_per_10k | peak_vram_bytes
Verdict: <adopt / reject / inconclusive> because <one line>
```

## 2026-11-14: project bootstrap
- Profiling gate (D4): nsys `--capture-range=cudaProfilerApi`, counts read from the `nsys stats`
  CSV reports `cuda_gpu_mem_size_sum` and `cuda_api_sum`. Rejected: hand-written SQL over the
  `--export=sqlite` DB, because nsys already computes these totals and SQL would add code volume.
- Framework candidates for Phase 2, all unproven on SM 6.1:
  1. JAX (`jit` + `vmap` + `lax.scan`). Pros: PureJaxRL-style, so env + policy + update can
     fuse into one XLA loop. Risk: XLA behaviour and shared-memory spills on Pascal as traffic
     branching grows.
  2. PyTorch (`torch.compile` + `torch.cuda.CUDAGraph`, optionally TorchRL). Pros: graph replay has near-zero
     dispatch cost on WSL2. Risk: graph breaks on resets and neighbour lookups.
  3. Custom kernels (Warp / Taichi / Triton / CUDA C++). Pros: spatial hashing, raycasts and
     graph kinematics map naturally to SIMT without padding. Risk: autodiff and NN integration.
- Toolchain constraint (D8): CUDA 13 dropped Pascal, so use CUDA 12.x framework builds and
  verify SM 6.1 kernels are present before adopting a wheel.
- Open follow-ups (from bootstrap review):
  - [Superseded by the 2026-11-14 CUPTI counter entry: nsys dropped.] nsys CSV format (quoting, preamble) is unverified because nsys is not installed yet. When it
    lands, run `gpu-sim bench` on a trivial fused workload, save one real `cuda_api_sum` and one
    `cuda_gpu_mem_size_sum` CSV under `tests/fixtures/`, and replace the `@pytest.mark.gpu`
    placeholder with that end-to-end run (Phase 1 done-criterion).
  - Risk for JAX: an XLA `lax.scan`/while loop may launch kernels from the host every iteration
    unless command buffers are enabled. Measure `launches_per_10k` early (BRIEF Q5).
  - [Superseded by the 2026-11-14 CUPTI counter entry.] Host-to-Host and peer copies are not counted as host copies; a SKIPPED nsys report (no
    memory ops) counts as 0, which is correct for the ideal case. Zero launches fails the gate.

## 2026-11-14: profiling gate moves from nsys to a CUPTI driver-API counter (D4)
- Evidence: nsys on this WSL2 + GTX 1070 records no CUDA data at all, so the nsys path was blind
  (every report empty, so the gate could only fail on zero launches).
- Replacement: `src/gpu_sim/cucount.c`, a CUPTI callback-API subscriber injected via
  `CUDA_INJECTION64_PATH`. It counts every driver-API call by name between cuProfilerStart and
  cuProfilerStop and writes `{name: count}` JSON at exit; `bench.py` builds it with gcc into
  `~/.cache/gpu_sim/` and classifies names into launches / host memcpys / host syncs.
- Probe results inside the capture range:
  - raw CUDA (`tests/fixtures/probe.cu`): 1000 launches, 3 H2D, 2 D2H, 4 syncs, all exact.
  - JAX: 101 `cuLaunchKernel`, 1 H2D, 1 D2H.
  - torch 2.7.1+cu126: 200 launches, 1 H2D, 1 D2H, syncs counted (2 `cuStreamSynchronize`,
    1 `cuCtxSynchronize`).
- `gpu-sim bench -- probe`: launches 1000, host_memcpys 5, host_syncs 4 (now the
  `@pytest.mark.gpu` end-to-end test).
- Known gap: XLA waits through `cuEventRecord` / `cuLaunchHostFunc` / `cuStreamWaitEvent`
  rather than `cu*Synchronize`, so `host_syncs` reads 0 for JAX. Its D2H readbacks are still
  counted as `host_memcpys`, which covers the C1 intent for telemetry readbacks.
- Generic `cuMemcpy`/`cuMemcpyAsync` (direction unknown under unified addressing) count as host
  copies, to be safe.
- Toolchain (D8): the latest torch wheel lacks sm_6x kernels ("no kernel image"), so pin
  `torch==2.7.1` from the cu126 index.
Verdict: adopt, because it gives exact counts for raw CUDA, JAX and PyTorch where nsys gives none.

## 2026-11-14: owner answers to BRIEF §10 Q1, Q2, Q4, Q5
- Q1: Stage 2 target is env-steps summed across all towns, ≥ 10^9/hour (~280k SPS), stretch.
  Batched steps/hour stays as the secondary normalising metric; `BENCH_RESULT` gains an
  optional `env_steps` field, reported as `env_steps_per_hour`.
- Q2: `CORE_BUDGET` = 2500, `TOTAL_BUDGET` = 6000, raised only at phase gates by the owner.
- Q4: Stage 1 is one population, P = 10,000 total.
- Q5: ≤ 100 launches/10k stays provisional; a Phase 1 sweep (steps fused per launch vs
  steps/hour) sets the C2 threshold.
