# gpu_sim: north star

Owner: Reuben. This file changes only by owner decision. Current numbers live in
[`DECISIONS.md`](../DECISIONS.md), work in [`BACKLOG.md`](BACKLOG.md), tooling in the code.

## Goal

A simulation and learning engine that runs **100% on the GPU**. It executes millions of
evolutionary generations or RL environment steps per hour on a single local GTX 1070. It starts
with a small co-evolution sanity check (hummingbird beaks vs flower geometry, Stage 1) and scales
into a **multi-agent town traffic RL simulation** (Stage 2). "Better" means more steps per hour
within the constraints below, with agents that still learn.

Phases, in gate order: 1 profiling harness, 2 Stage 1 framework shootout, 3 traffic environment
(built with the shootout winner), 4 end-to-end GPU RL (Stage 2 target). The next gate is
in BACKLOG.md.

## Hard constraints (never cut; each is testable)

| # | Invariant | How it is checked |
|---|---|---|
| C1 | **GPU residency, no CPU syncs.** The env step (kinematics, collisions, rewards, resets) and the policy/update loop both run in VRAM. Inside the timed loop, host↔device copies ≤ `steps // 100_000` and host sync calls ≤ `steps // 100_000`: telemetry is packed into one device buffer and read back at most once per 10^5 steps. | `uv run gpu-sim bench -- <workload>`: `host_memcpys`, `host_syncs` |
| C2 | **Fused launches.** The CPU only launches compiled loops that fuse thousands of steps (each launch is costly under WSL2): ≤ 100 host-side kernel/graph launches per 10,000 steps (provisional; set by the steps-fused-per-launch sweep in BACKLOG). | same, `launches_per_10k` |
| C3 | **VRAM.** Peak device memory used by the process (including the CUDA context and framework pools, not an allocator counter) < 7.0e9 bytes, leaving the rest of the 8 GB for WSL2/OS. | same, `peak_vram_bytes` |
| C4 | **Hardware.** The NVIDIA GTX 1070 (Pascal SM 6.1, 8 GB, 1920 cores, ~256 GB/s) on WSL2. $0 cloud. | `nvidia-smi --query-gpu=name,compute_cap --format=csv` |
| C5 | **FP32 and integers only.** No fp16/bf16 or autocast (Pascal runs FP16 at 1:64). fp32 for kinematics and weights; int32/int16/int8/uint8 for indices and discrete states. | code review; `grep -rniE "float16\|bfloat16\|autocast" src/` is empty |
| C6 | **Fixed shapes.** No dynamic shapes in the step loop: no boolean-mask indexing, growing lists or host `if` on tensor values; padded tensors and `where(cond, new, old)` instead. | zero recompiles inside the timed loop (framework counter, see DECISIONS.md); code review |
| C7 | **Evidence-gated pivots.** Before proposing a framework or memory-layout change, record its `gpu-sim bench` numbers (median of 3 runs) as the evidence for that decision in `DECISIONS.md`. | the decision's evidence column in `DECISIONS.md` |

The C1 to C3 thresholds are also constants in `src/gpu_sim/bench.py`; `tests/test_bench.py`
asserts they match this table, so change both together.

## Ranked requirements (under pressure, cut from the bottom)

1. GPU residency (C1, C2)
2. VRAM ceiling (C3)
3. Throughput targets
4. Scaling from the sanity check to traffic RL
5. Learning quality (the agents actually learn)
6. Code simplicity and framework familiarity

## Targets

A "step" (`steps` in `BENCH_RESULT`) is one batched step that advances every parallel env (or
the population) at once; C1 and C2 count per batched step. Stage 1: `steps` = generations.
Stage 2 workloads also report `env_steps` (batched steps × N towns), which `gpu-sim bench` turns
into `env_steps_per_hour`; batched steps/hour stays a secondary, normalising metric. The latest
measured values are in `DECISIONS.md`.

| Measure | Target | Command | Run tier |
|---|---|---|---|
| Stage 1 throughput: generations/hour, one population, P = 10,000 total | ≥ 10^6 (~280 gen/s) | `uv run gpu-sim bench -- <stage-1 workload>` | GPU, local; 10^5–10^6 generations |
| Stage 2 throughput: env-steps/hour summed across all N towns | ≥ 10^9 (~280k SPS); stretch (owner, 2026-11-14) | same, traffic workload | GPU, local |
| Stage 1 learning quality (proposed, owner to confirm) | trait matching (beak vs flower geometry) improves over generations versus a no-selection control run, and the GPU result matches a small CPU reference implementation | workload + pytest | GPU, local |
| Stage 2 learning quality (proposed, owner to confirm) | the learned policy beats (a) a random policy and (b) a hand-coded fixed-timer traffic-light baseline on mean trip time and collisions, over fixed evaluation seeds | evaluation workload | GPU, local |
| C1 to C3 | as above | `gpu-sim bench`, every run | GPU, local |
| Unit tests + lint | green | `uv run pytest -q && uv run ruff check . && uv run ruff format --check .` | CPU; local + CI |

Exact learning-quality margins are set after the first runs, as backlog items.

Noise band (proposed): a throughput change smaller than ±5% across 3 runs is noise. Claim a
speedup only from the median of 3 runs.

## Budgets

- **Code volume** (`uv run pytest tests/test_code_volume.py`; table via
  `uv run python scripts/code_volume.py`): AST nodes, with `CORE_BUDGET` for `src/gpu_sim/` and
  the larger `TOTAL_BUDGET` for all project Python. The values are the constants in
  `scripts/code_volume.py`, set by the owner and raised only at phase gates. Names, comments,
  docstrings, type hints and imports are free. Never game it (golfing, packed lines, cryptic
  names). When it binds, simplify, reuse a library, or ask the owner at the start of a session.
  Never raise it mid-session.
- **Money:** $0. No cloud GPUs and no paid services. Any spend needs the owner.
- **Time:** unit suite < 30 s on CPU. A gate `bench` run ≤ 10 min wall (proposed). CI (CPU
  only) < 5 min.

## Settled principles (do not re-litigate; flag concerns to the owner)

- Local GTX 1070 on WSL2 is the only target, at $0. Why: owner hardware and cost.
- FP32 and integer arithmetic only (C5). Why: Pascal FP16 runs at 1:64.
- Fixed-shape, branchless or masked step functions (C6). Why: dynamic shapes break fusion and force recompiles.
- Pivots are evidence-gated (C7); the framework is chosen by measurement, not preference.
- GPU gates run locally; GitHub CI is CPU-only. Why: no GPU runners at $0.

Technical decisions (tooling, pins, framework status) live in `DECISIONS.md`.

## Backlog prioritisation

Every [`BACKLOG.md`](BACKLOG.md) item serves a requirement or target above, or unblocks one
that does. Order: blocking first, then the rank of what it serves, then cheapest. Full rules are
in the BACKLOG preamble.

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
- spend any money
- run `sudo` or install system packages (CUDA toolkit)
- push to `main` on the public GitHub remote, or change repo settings or visibility
- delete untracked data

## Non-goals

- Cloud or multi-GPU training, and other GPUs as targets (newer cards may run it, but nothing is tuned for them)
- Rendering or a GUI. Verification is CLI-only, though offline plots from telemetry are fine.
- Realistic traffic fidelity beyond what the RL task needs. Use IDM or bicycle approximations.
- FP16, bf16, tensor-core or mixed-precision paths
- A general-purpose RL library. Build only what the two stages need.

## Open owner questions

Agents add questions here instead of guessing.

- Q6. Confirm the "(proposed)" items: the learning-quality targets, the ±5% noise band and the
  10-minute gate bench run.

## Freshness

Last reviewed 2026-11-14 by the owner. Re-review at each phase gate.
