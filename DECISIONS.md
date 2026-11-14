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
  - nsys CSV format (quoting, preamble) is unverified because nsys is not installed yet. When it
    lands, run `gpu-sim bench` on a trivial fused workload, save one real `cuda_api_sum` and one
    `cuda_gpu_mem_size_sum` CSV under `tests/fixtures/`, and replace the `@pytest.mark.gpu`
    placeholder with that end-to-end run (Phase 1 done-criterion).
  - Risk for JAX: an XLA `lax.scan`/while loop may launch kernels from the host every iteration
    unless command buffers are enabled. Measure `launches_per_10k` early (BRIEF Q5).
  - Host-to-Host and peer copies are not counted as host copies; a SKIPPED nsys report (no
    memory ops) counts as 0, which is correct for the ideal case. Zero launches fails the gate.
