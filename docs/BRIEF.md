# gpu_sim: north star

Owner: Reuben. This file changes only by owner decision. Current numbers live in
[`DECISIONS.md`](../DECISIONS.md), work in [`BACKLOG.md`](BACKLOG.md), tooling in the code.

## Goal

A **convincing**, **clean** showcase, for the owner and others, that simulation and learning can
run 100% on **this GPU** (a single local GTX 1070). Each word sets a principle the rest of this
brief applies:

- **convincing**: every claim is judged against a known baseline (Evidence principle);
- **clean**: the simplest design that makes the point (requirement 3, Budgets);
- **this GPU**: work with the machine's real costs (Hard constraints).

Horizon: if it works, the owner may scale it to larger GPUs and more sophisticated simulations,
towards research. That is not in scope now (Non-goals), but when two options are equally simple,
prefer the one that doesn't depend on the GTX 1070.

The engine executes millions of evolutionary generations or RL environment steps per hour. It starts
with a small co-evolution sanity check (hummingbird beaks vs flower geometry, Stage 1) and scales
into a **multi-agent town traffic RL simulation** (Stage 2). "Better" means a larger speedup over
a CPU reference on the same results, within the constraints below, with learning that beats a
null control.

Phases, in gate order: 1 profiling harness, 2 Stage 1 framework shootout, 3 traffic environment
(built with the shootout winner, auto-resetting), 4 end-to-end GPU RL (Stage 2 target). The
next gate is in BACKLOG.md.

## Evidence principle: compare against a known baseline

Every claim is a comparison against a baseline that is simpler and better understood than the
thing claimed about. A number with no baseline is not evidence. The baseline runs the same
workload on the same seeds through the same harness, and it exists before the candidate does.

| Claim | Baseline | The claim holds when |
|---|---|---|
| Correct | a small CPU (NumPy) reference implementation of the workload | the GPU result matches it within a stated tolerance |
| Fast | that same CPU reference, and the previous best median in DECISIONS | speedup and steps/hour are reported against both |
| Learns | a null control (no selection; a random policy), then a simple heuristic (e.g. a fixed-timer traffic light) | it beats each by more than the spread across fixed seeds |
| Instrument is right | a known-answer workload (e.g. `tests/fixtures/probe.cu`), and the same run without the instrument | counts are exact and overhead is inside the noise band |
| Change is real | the baseline's own repeated runs | the median of 3 moves by more than their spread (floor ±5%, proposed) |

A speed claim counts only for a workload that passes "Correct". Throughput is not monotonic
(owner, 2026-11-14), so a parameter's best value is measured by a sweep, not assumed. The sweep
list is in the BACKLOG preamble.

## Hard constraints (never cut; each is testable)

| # | Invariant | How it is checked |
|---|---|---|
| C1 | **GPU residency, no CPU syncs.** The env step (kinematics, collisions, rewards, resets) and the policy/update loop both run in VRAM. Inside the timed loop, host↔device copies ≤ `steps // 100_000` and host sync calls ≤ `steps // 100_000`: telemetry is packed into one device buffer and read back at most once per 10^5 steps. | `uv run gpu-sim bench -- <workload>`: `host_memcpys`, `host_syncs` |
| C2 | **Fused launches.** The CPU only launches compiled loops that fuse thousands of steps (each launch is costly under WSL2): ≤ 100 host-side kernel/graph launches per 10,000 steps (provisional; set by the steps-fused-per-launch sweep table in DECISIONS). | same, `launches_per_10k` |
| C3 | **VRAM.** Peak device memory used by the process (including the CUDA context and framework pools, not an allocator counter) < 7.0e9 bytes, leaving the rest of the 8 GB for WSL2/OS. | same, `peak_vram_bytes` |
| C4 | **Hardware.** The NVIDIA GTX 1070 (Pascal SM 6.1, 8 GB, 1920 cores, ~256 GB/s) on WSL2. $0 cloud. Why: owner hardware and cost. | `nvidia-smi --query-gpu=name,compute_cap --format=csv` |
| C5 | **FP32 and integers only.** No fp16/bf16 or autocast. Why: FP16 math runs 64× slower than FP32 on this card (per Nvidia's GP104 spec, not measured here). fp32 for kinematics and weights; int32/int16/int8/uint8 for indices and discrete states. | code review; `tests/test_constraints.py` (no fp16/bf16/autocast in `src/`) |
| C6 | **Fixed shapes.** No dynamic shapes in the step loop: no boolean-mask indexing, growing lists or host `if` on tensor values; padded tensors and `where(cond, new, old)` instead. Why: dynamic shapes break fusion and force recompiles. | zero recompiles inside the timed loop (framework counter, see DECISIONS.md); code review |
| C7 | **Evidence-gated pivots.** Before proposing a framework or memory-layout change, record its `gpu-sim bench` numbers (median of 3 runs) as the evidence for that decision in `DECISIONS.md`. | the decision's evidence column in `DECISIONS.md` |

The C1 to C3 thresholds are also constants in `src/gpu_sim/bench.py`; `tests/test_bench.py`
asserts they match this table, so change both together.

## Ranked requirements (under pressure, cut from the bottom)

1. GPU residency (C1, C2)
2. VRAM ceiling (C3)
3. Simplicity over the last 1% of speed. Complexity is the measure; the code-volume budget is only a proxy and a ceiling, and
   sometimes more code is simpler. Complexity must buy a gain well beyond the noise band (not
   +1% for 10× the complexity); this is judgement, not a formula. When meeting a target would
   take a large complexity increase, it is the owner's call: put it in Open owner questions.
4. Throughput targets
5. Scaling from the sanity check to traffic RL
6. Learning quality (beats its null control and heuristic baselines; see Evidence principle)
7. Framework familiarity

## Targets

A "step" (`steps` in `BENCH_RESULT`) is one batched step that advances every parallel env (or
the population) at once; C1 and C2 count per batched step. Stage 1: `steps` = generations.
Stage 2 workloads also report `env_steps` (batched steps × N towns), which `gpu-sim bench` turns
into `env_steps_per_hour`; batched steps/hour stays a secondary, normalising metric. The latest
measured values are in `DECISIONS.md`.

| Measure | Baseline | Target | Command | Run tier |
|---|---|---|---|---|
| Stage 1 throughput: generations/hour, one population, P = 10,000 total | Stage 1 CPU reference | ≥ 10^6 (~280 gen/s) | `uv run gpu-sim bench -- <stage-1 workload>` | GPU, local; 10^5–10^6 generations |
| Stage 2 throughput: env-steps/hour summed across all N towns | Stage 2 CPU reference | ≥ 10^9 (~280k SPS); stretch (owner, 2026-11-14) | same, traffic workload | GPU, local |
| C1 to C3 | as above | as above | `gpu-sim bench`, every run | GPU, local |
| Unit tests + lint | n/a | green | `uv run pytest -q && uv run ruff check . && uv run ruff format --check .` | CPU; local + CI |

Correctness and learning are judged by the Evidence principle table and applied as gate criteria
in BACKLOG.md.

## Budgets

- **Code volume:** AST-node budgets for core (`src/gpu_sim/`) and total (all project Python),
  set by the owner in `scripts/code_volume.py`. Check with `uv run pytest tests/test_code_volume.py`
  (table: `uv run python scripts/code_volume.py`). Names, comments, docstrings, type hints and
  imports are free. Never game it. The budget is backpressure, not a limit to negotiate: when it
  binds, find a more elegant solution (simplify, reuse a library), or push back on the
  requirement driving it (see "Push back early"). Only the owner changes it.
- **Money:** $0. No cloud GPUs and no paid services. Any spend needs the owner.
- **Time:** unit suite < 30 s on CPU. A gate `bench` run ≤ 10 min wall (proposed). CI (CPU
  only) < 5 min.

## Settled decisions

The hard constraints are settled. Technical decisions, including the owner-gated pins, live in
[`DECISIONS.md`](../DECISIONS.md). Do not re-litigate them; flag concerns to the owner.

## Autonomy boundary

**Agents may, without asking:** do anything that moves a target without breaking a constraint
or budget. That covers writing code and tests, adding uv dependencies, running local benches,
committing locally (including on `main`), pushing non-`main` branches and opening PRs,
re-sorting the backlog, and switching frameworks or layouts when `DECISIONS.md` holds the bench
evidence (C7).

**Agents need the owner to:**
- change this brief or a settled principle
- cut a requirement or lower a target
- raise a code-volume budget
- accept a large complexity increase to meet a target (rank 3)
- change an owner-gated decision in DECISIONS.md
- spend any money
- run `sudo` or install system packages (CUDA toolkit)
- push to `main` on the public GitHub remote, or change repo settings or visibility
- delete untracked data

Push back early: if you see no elegant path to a requirement or target within the budgets, push
back on the requirement at the start of a session, before building. Don't force an inelegant
path, and don't ask for more budget to fit one.

## Non-goals

- Cloud or multi-GPU training, and other GPUs as targets, for now (see Goal, Horizon)
- A live GUI or rendering inside the timed loop. Visuals for the showcase are welcome, but they
  must be generic across both stages, small, and add no overhead to benchmarked runs (e.g.
  rendered offline from a seeded replay). Verification stays CLI-only.
- Realistic traffic fidelity beyond what the RL task needs. Use IDM or bicycle approximations.
- A general-purpose RL library. Build only what the two stages need.

## Open owner questions

Agents add questions here instead of guessing.

- Q6. Confirm the "(proposed)" items: the ±5% noise-band floor and the 10-minute gate bench run.

## Freshness

Last reviewed 2026-11-14 by the owner. Re-review at each phase gate.
