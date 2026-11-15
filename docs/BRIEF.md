# gpu_sim: north star

Owner: Reuben. Changes need the owner, except (measured) numbers (P2). Current numbers live in
[`DECISIONS.md`](../DECISIONS.md), work and the phase gates in [`BACKLOG.md`](BACKLOG.md),
tooling in the code.

## Goal

A **convincing**, **clean** showcase, for the owner and others, that simulation and learning can
run 100% on **this GPU** (a single local GTX 1070): **convincing** is P1 and P2, **clean** is P4,
**this GPU** is P3.

The engine runs millions of evolutionary generations or RL environment steps per hour. It starts
with a small co-evolution sanity check (hummingbird beaks vs flower geometry, Stage 1) and scales
into a **multi-agent town traffic RL simulation** (Stage 2).

Horizon: if it works, the owner may scale it to larger GPUs and more sophisticated simulations,
towards research. That is not in scope now (Non-goals), but when two options are equally simple,
prefer the one that doesn't depend on the GTX 1070.

## Principles

**P1 Baseline.** Every claim is a comparison with something simpler and better understood, built
first, run through the same harness on the same seeds. Speed: a CPU reference. Learning: a null
control (no selection, random policy), then a heuristic. Instruments: a known answer. Change: the
baseline's own spread. A result that fails its baseline is not a result. The ranked requirements
trade off only among valid results. It generates:

| Claim | Baseline | The claim holds when |
|---|---|---|
| Correct | a small CPU (NumPy) reference implementation of the workload | the GPU result matches it within a stated tolerance |
| Fast | that same CPU reference, and the previous best median in DECISIONS | speedup and steps/hour are reported against both; only a workload that is Correct counts |
| Learns | a null control, then a simple per-stage heuristic | it beats each by more than the spread across fixed seeds |
| Instrument is right | a known-answer workload, and the same run without the instrument | counts are exact and overhead is inside the noise band |
| Change is real | the baseline's own repeated runs | the median of 3 moves by more than their spread (floor ±5%, proposed) |

Pivots: before proposing a framework or memory-layout change, record its `gpu-sim bench` numbers
(median of 3) in DECISIONS as that decision's evidence. The per-stage baselines are in DECISIONS.

**P2 Measure, don't assume.** A parameter's value is measured, not chosen. Every number is tagged
(owner) or (measured: <sweep/evidence in DECISIONS>). Agents update (measured) numbers in BRIEF
and code together; (owner) numbers change only with the owner. Throughput is not monotonic (owner,
2026-11-14), so a parameter's best value comes from a sweep; the sweep list is in the BACKLOG
preamble.

**P3 Pay the machine's real costs.** Work stays where it is cheapest on this card. Every host
interaction, recompile or slow dtype is a counted cost with a ceiling. It generates C1 to C6.

**P4 Complexity must pay rent.** Add complexity only for a gain well beyond noise (not +1% for
10× the complexity; judgement, not a formula). Code volume is the ceiling, and only a proxy:
sometimes more code is simpler. When it binds, find a simpler design or push back on the
requirement. A large increase, meaning more than 25% of the remaining core budget or a new
language or toolchain (owner, 2026-11-14), needs the owner: put it in Open owner questions.

Push back early: if you see no elegant path to a requirement or target within the budgets, push
back on the requirement at the start of a session, before building. Don't force an inelegant
path, and don't ask for more budget to fit one.

**P5 Reversible is autonomous.** Agents act freely on what git can undo; the owner gates what
can't be undone, costs money, or changes the goal.

*Agents may, without asking:* write code and tests, add uv dependencies, run local benches,
commit locally (including on `main`), push non-`main` branches and open PRs, re-sort the backlog,
update (measured) numbers, and switch frameworks or layouts once DECISIONS holds the evidence (P1).

*Agents need the owner to:*
- change this brief's goal, a principle or an (owner) number
- cut a requirement or lower a target
- raise a code-volume budget
- accept a large complexity increase (P4)
- change an owner-gated decision in DECISIONS
- spend any money
- run `sudo` or install system packages (CUDA toolkit)
- push to `main` on the public GitHub remote, or change repo settings or visibility
- delete untracked data

## Hard constraints (never cut; each is testable)

| # | Invariant | Source | How it is checked |
|---|---|---|---|
| C1 | **GPU residency, no CPU syncs.** The env step (kinematics, collisions, rewards, resets) and the policy/update loop both run in VRAM. Inside the timed loop, host↔device copies ≤ `steps // 100_000` and host sync calls ≤ `steps // 100_000`: telemetry is packed into one device buffer and read back at most once per 10^5 steps. | P3 (owner) | `uv run gpu-sim bench -- <workload>`: `host_memcpys`, `host_syncs` |
| C2 | **Fused launches.** The CPU only launches compiled loops that fuse many steps (each launch is costly under WSL2): ≤ 100 host-side kernel/graph launches per 10,000 steps. | P3 (measured: the steps-fused-per-launch sweep; provisional until it runs) | same, `launches_per_10k` |
| C3 | **VRAM.** Peak device memory used by the process (including the CUDA context and framework pools, not an allocator counter) < 7.0e9 bytes, leaving the rest of the 8 GB for WSL2/OS. | P3 (owner) | same, `peak_vram_bytes` |
| C4 | **Hardware.** The GTX 1070 on WSL2, $0 cloud; specs in DECISIONS. | P3 (owner) | `nvidia-smi --query-gpu=name,compute_cap --format=csv` |
| C5 | **FP32 and integers only.** No fp16/bf16 or autocast (FP16 is far slower on this card; DECISIONS). fp32 for kinematics and weights; int32/int16/int8/uint8 for indices and discrete states. | P3 (owner) | code review; `tests/test_constraints.py` |
| C6 | **Fixed shapes.** No boolean-mask indexing, growing lists or host `if` on tensor values in the step loop; padded tensors and `where(cond, new, old)` instead, because dynamic shapes break fusion and force recompiles. | P3 (owner) | zero recompiles inside the timed loop (DECISIONS); code review |

The C1 to C3 thresholds are also constants in `src/gpu_sim/bench.py`; `tests/test_bench.py`
asserts they match this table, so change both together.

## Ranked requirements (under pressure, cut from the bottom)

1. Simplicity over the last 1% of speed (P4)
2. Throughput targets
3. Scaling from the sanity check to traffic RL
4. Showcase artefacts
5. Framework familiarity

Not ranked: residency and VRAM (C1 to C3) are invariants, never cut; correctness and learning are
validity conditions (P1).

## Targets

The step definition and the `BENCH_RESULT` fields are in the `src/gpu_sim/bench.py` docstring;
each gate's population size and run length are in its BACKLOG gate. Latest values: DECISIONS.

| Measure | Baseline | Target | Gate |
|---|---|---|---|
| Stage 1 throughput: generations/hour | Stage 1 CPU reference | ≥ 10^6 (~280 gen/s) (owner) | the Stage 1 gate |
| Stage 2 throughput: env-steps/hour summed across all towns | Stage 2 CPU reference | 10^9 (~280k SPS): ASPIRATIONAL (owner, 2026-11-14); best measured value in DECISIONS | the Stage 2 gate is a speedup over this baseline (median of 3 above 1× by more than its spread, P1) with C1 to C3 passing |
| C1 to C3 | as above | as above (owner and measured, per row) | every `gpu-sim bench` run |
| Unit tests + lint | n/a | green, local and CI (owner) | every commit |

## Budgets

- **Code volume (owner):** AST-node budgets for core (`src/gpu_sim/`) and total (all project
  Python) in `scripts/code_volume.py`, checked by `tests/test_code_volume.py` (what counts:
  CLAUDE.md). Never game it; it is backpressure, not a limit to negotiate (P4). Only the owner
  changes it.
- **Money (owner):** $0. No cloud GPUs and no paid services. Any spend needs the owner.
- **Time (owner):** unit suite < 30 s on CPU; CI (CPU only) < 5 min; a gate `bench` run ≤ 10 min
  wall (proposed).

## Settled decisions

The hard constraints are settled. Technical decisions, including the owner-gated pins, live in
[`DECISIONS.md`](../DECISIONS.md). Do not re-litigate them; flag concerns to the owner.

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
