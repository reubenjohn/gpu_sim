# gpu_sim: north star

Owner: Reuben. Last reviewed 2026-11-14; re-review at each phase gate (§7). Edit only with the owner's approval. Current numbers live
in the commands and in `DECISIONS.md`, not here.

## 1. Goal

A simulation and learning engine that runs **100% on the GPU**. It executes millions of
evolutionary generations or RL environment steps per hour on a single local GTX 1070. It starts
with a small co-evolution sanity check (hummingbird beaks vs flower geometry) and scales into a
**multi-agent town traffic RL simulation**. "Better" means more steps per hour within the
constraints below, with agents that still learn.

## 2. Hard constraints (never cut; each is testable)

| # | Invariant | How it is checked |
|---|---|---|
| C1 | **GPU residency, no CPU syncs.** The env step (kinematics, collisions, rewards, resets) and the policy/update loop (mutation, or rollout + PPO/DQN update) both run in VRAM. Inside the timed loop, host↔device copies ≤ `steps // 100_000` and host sync calls (`cuda*/cu*Synchronize`) ≤ `steps // 100_000`, so telemetry is read back at most once per 10^5 steps. Pack telemetry into one device buffer: one readback = one D2H copy. | `uv run gpu-sim bench -- <workload>`: `host_memcpys`, `host_syncs` |
| C2 | **Fused launches.** The CPU only launches a compiled megakernel, CUDA Graph or XLA loop, fusing thousands of steps per launch (WSL2/WDDM makes each launch costly): ≤ 100 host-side kernel/graph launches per 10,000 steps (provisional; set by the Phase 1 sweep, §7). | same, `launches_per_10k` |
| C3 | **VRAM.** Peak device memory used (NVML, including the CUDA context and framework pools, not an allocator counter) < 7.0e9 bytes, leaving the rest of the 8 GB for WSL2/OS. JAX workloads set `XLA_PYTHON_CLIENT_PREALLOCATE=false`. | same, `peak_vram_bytes` |
| C4 | **Hardware.** Targets the NVIDIA GTX 1070 (Pascal SM 6.1, 8 GB, 1920 cores, ~256 GB/s) on WSL2. $0 cloud. | `nvidia-smi --query-gpu=name,compute_cap --format=csv` |
| C5 | **FP32 and integers only.** No fp16/bf16 or autocast (Pascal runs FP16 at 1:64). Use fp32 for kinematics and weights, int32/int16/int8/uint8 for indices and discrete states. | code review; `grep -rniE "float16\|bfloat16\|autocast" src/` is empty |
| C6 | **Fixed shapes.** No dynamic shapes in the step loop: no `tensor[mask]`, no growing lists, no host `if` on tensor values. Use padded tensors and `where(cond, new, old)`. | zero recompiles inside the profiled range (`jax.log_compiles` / the `torch._dynamo` recompile counter); code review |
| C7 | **Evidence-gated pivots.** Log a framework or memory-layout experiment's `gpu-sim bench` numbers (median of 3 runs) in `DECISIONS.md` *before* proposing the change. | `DECISIONS.md` entry per pivot |

The thresholds in C1 to C3 are also constants in `src/gpu_sim/bench.py`. `tests/test_bench.py`
asserts they match this table, so change both together.

## 3. Ranked requirements (under pressure, cut from the bottom)

1. GPU residency (C1, C2)
2. VRAM ceiling (C3)
3. Throughput targets (§4)
4. Scaling from the sanity check to traffic RL
5. Learning quality (the agents actually learn)
6. Code simplicity and framework familiarity

## 4. Targets

A "step" (`steps` in `BENCH_RESULT`) is one batched step that advances every parallel env (or
the population) at once; C1 and C2 are counted per batched step. For Stage 1, `steps` =
generations. Stage 2 workloads also report `env_steps`: env-steps summed across all towns
(batched steps × N), which `gpu-sim bench` turns into `env_steps_per_hour`. The Stage 2 target
is in summed env-steps; batched steps/hour is reported as a secondary, normalising metric.
`steps_per_hour` comes from `gpu-sim bench`. Fill in "now" from `DECISIONS.md` (the latest
entry per row).

| Measure | Now | Target | Command | Run tier |
|---|---|---|---|---|
| Stage 1: hummingbird generations/hour (one population, P = 10,000 total) | not measured | ≥ 1,000,000 (~280 gen/s) | `uv run gpu-sim bench -- <stage-1 workload>` | GPU, local; 10^5–10^6 generations |
| Stage 2: traffic env-steps/hour summed across all N parallel towns (`env_steps_per_hour`) | not measured | ≥ 10^9 summed env-steps (~280k SPS); stretch (owner, 2026-11-14) | same, traffic workload | GPU, local |
| Host memcpys / host syncs in the timed loop | n/a | ≤ steps // 100,000 each | same | each bench run |
| Launches per 10k steps | n/a | ≤ 100 | same | each bench run |
| Peak VRAM | n/a | < 7.0e9 bytes | same | each bench run |
| Unit tests + lint | see CI | green | `uv run pytest -q && uv run ruff check . && uv run ruff format --check .` | CPU; local + CI |

Noise band (proposed): a throughput change smaller than ±5% across 3 runs is noise. Claim a
speedup only from the median of 3 runs.

## 5. Budgets

- **Code volume** (`uv run pytest tests/test_code_volume.py`, table via
  `uv run python scripts/code_volume.py`): counts AST nodes. Names, comments, docstrings, type
  hints and imports are free. Two budgets: `CORE_BUDGET` covers `src/gpu_sim/`, and the much larger
  `TOTAL_BUDGET` covers all project Python, tests included. Both are set by the owner (values in the script:
  `CORE_BUDGET` = 2500, `TOTAL_BUDGET` = 6000, owner, 2026-11-14). They start small for constant
  back-pressure and are raised only at phase gates, by the owner. Code volume only approximates complexity, so never game it (golfing,
  packed lines, cryptic names). When it binds, simplify, reuse a library instead of hand-rolling
  (e.g. use CUPTI callbacks rather than hand-parsing profiler databases), or ask the
  owner at the start of a session to raise it. Never raise it mid-session.
- **Money:** $0. No cloud GPUs and no paid services. Spend of any amount needs the owner.
- **Time:** unit suite < 30 s on CPU. A gate `bench` run ≤ 10 min wall (proposed). CI (CPU only)
  < 5 min.

## 6. Settled decisions (do not re-litigate; flag concerns to the owner)

Full rationale and the benchmark log: [`DECISIONS.md`](../DECISIONS.md).

- D1. Local GTX 1070 on WSL2 is the only target; $0 cloud. Why: owner hardware and cost.
- D2. FP32 and integer arithmetic only (C5). Why: Pascal FP16 runs at 1:64.
- D3. Fixed-shape, branchless or masked step functions (C6). Why: dynamic shapes break fusion and force recompiles.
- D4. The profiling gate injects a CUPTI driver-API callback counter (`src/gpu_sim/cucount.c`, via `CUDA_INJECTION64_PATH`) that counts calls by name between cudaProfilerStart and cudaProfilerStop. Why: nsys records no CUDA data on WSL2 + Pascal; CUPTI driver-API callbacks do. The workload contract is in `src/gpu_sim/bench.py`.
- D5. Python 3.12 managed by uv (`uv run`, `uv add`). No pip and no system Python.
- D6. GPU gates run locally. GitHub CI is CPU-only (lint, unit tests, code volume); `@pytest.mark.gpu` tests skip without the GPU, nvcc or gcc.
- D7. **The framework is NOT settled.** JAX, PyTorch (CUDA Graphs) and custom kernels (Warp/Taichi/Triton/CUDA) are hypotheses. The Phase 2 shootout picks one, with evidence in `DECISIONS.md` (C7). Hybrids (custom env kernels + a JAX/PyTorch policy) are valid candidates. A candidate with no working SM 6.1 build is dropped from the shootout, and the failed check is logged in `DECISIONS.md`. No owner needed.
- D8. GPU frameworks must ship CUDA 12.x builds that support SM 6.1. CUDA 13 dropped Pascal. Check SM 6.1 support for any wheel before adopting it. PyTorch: pin `torch==2.7.1` from the cu126 index; the latest wheel lacks sm_6x kernels ("no kernel image"). Do not upgrade the Windows driver past the R580 branch, the last to support Pascal (the installed driver is 581.29).

## 7. Roadmap (gates, in order)

1. **Phase 1, profiling harness**: `gpu-sim bench` reports host memcpys, launches per 10k steps,
   steps/hour and peak VRAM, and gates on C1 to C3. Done when it runs end to end on the GTX 1070
   against a trivial fused workload. One-time sweep: steps fused per launch vs steps/hour on the
   GTX 1070; record the curve concisely in DECISIONS.md and set C2's threshold from it.
2. **Phase 2, hummingbird co-evolution shootout**: a vectorized loop (P = 10,000; beak morphology
   vs flower geometry) in JAX, PyTorch with CUDA Graphs, and one custom-kernel backend. Each runs
   10^5 to 10^6 generations back to back in VRAM. Gate: ≥ 10^6 gen/hour with C1 to C3 passing.
   Log the winner in `DECISIONS.md`.
3. **Phase 3, traffic minimal viable environment** (built with the Phase 2 winner): fixed tensors `[envs, max_vehicles, state]`
   and `[envs, max_intersections, light_state]`. Branchless IDM or bicycle kinematics,
   intersection conflict resolution, and auto-reset by masking. Gate: no recompiles and no CPU
   fallbacks in the timed loop.
4. **Phase 4, end-to-end GPU RL**: an FP32 actor-critic MLP and a `scan`-style rollout fused with
   the env. Report SPS and episodes/hour. Gate: Stage 2 target with C1 to C3 passing.

Open design variables, decided by evidence in `DECISIONS.md` (all must avoid dynamic allocation): the topology (2D grid with spatial
hashing vs a lane graph as fixed adjacency tensors), the observation (kinematic feature vectors
vs local occupancy grids, which must not need CNN rendering in VRAM), and the learner (neuroevolution with GA/CMA-ES vs PPO on the GPU).

### Sweeps (owner, 2026-11-14)

Throughput is not monotonic: a smaller batch can beat a larger one once state spills out of
cache. Measure, don't assume. Each sweep runs once per framework on the GTX 1070, and its
result goes into `DECISIONS.md` as a short table plus a one-line takeaway, so later agents
reuse it instead of re-running it. Re-run a sweep only when the framework or memory layout changes.

| Sweep | When | Notes |
|---|---|---|
| Batch size (population P, towns N) | required, Phase 2 and Phase 3 | find the peak and where it falls off; check C3 at the peak |
| Steps fused per launch | required, Phase 1 (sets C2) and per framework | also watch compile time and VRAM |
| Counter overhead | once, Phase 1 | the same workload's steps/hour with and without `gpu-sim bench` (CUPTI); expected ≈ 0 |
| Telemetry interval | once per framework | 10^4 to 10^6 steps; C1 caps it at ≥ 10^5 |
| Memory layout (array-of-structs vs struct-of-arrays) | once per framework | often 2–5× on Pascal |
| Integer width (int32 / int16 / int8) for indices and states | once per framework | narrower saves bandwidth; some ops are slower |
| Threads per block | custom kernels only | skip for JAX/PyTorch |

## 8. Autonomy boundary

**Agents may, without asking:** do anything that moves a target without breaking a constraint
or budget. That covers writing code and tests, adding uv dependencies, running local benches,
committing locally (including on `main`), pushing non-`main` branches and opening PRs, and switching frameworks or layouts
when `DECISIONS.md` holds the bench evidence (C7).

**Agents need the owner to:**
- change this brief or a settled decision
- cut a requirement or lower a target
- raise a code-volume budget
- spend any money
- run `sudo` or install system packages (CUDA toolkit)
- push to `main` on the public GitHub remote, or change repo settings or visibility
- delete untracked data

## 9. Non-goals

- Cloud or multi-GPU training, and other GPUs as targets (newer cards may run it, but nothing is tuned for them)
- Rendering or a GUI. Verification is CLI-only, though offline plots from telemetry are fine.
- Realistic traffic fidelity beyond what the RL task needs. Use IDM or bicycle approximations.
- FP16, bf16, tensor-core or mixed-precision paths
- A general-purpose RL library. Build only what the two stages need.

## 10. Open owner questions

Agents add questions here instead of guessing.

- Q3. Learning-quality bar for Phase 4: what reward or behaviour counts as "learns"? (Undecided.)
