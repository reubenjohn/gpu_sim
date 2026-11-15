# Backlog

Pick the top unblocked item. Only the orchestrating agent edits this file; subagents report
items upward.

Rules:
1. Every item names the [BRIEF](BRIEF.md) requirement or target it serves, or "unblocks B<n>"
   (it then inherits that item's priority). An item serving nothing is dropped.
2. Order: blocking items first, then the BRIEF rank of what it serves, then the cheapest first.
   Agents re-sort without asking.
3. Each item has a checkable "done when".
4. Finished items are DELETED, not checked off: git history keeps them and
   [DECISIONS.md](../DECISIONS.md) keeps results and evidence.
5. Only the NEXT milestone is a gate item, with its work items under it. Later phases are not
   written out until they are next; their gate criteria apply the BRIEF constraints, end
   targets and P1 Baseline (each claim against its baseline). Every gate item's
   done-when includes the owner re-reviewing the BRIEF.
6. At most ~15 active items; overflow goes to "Parked" at the bottom. Cap: 10 Parked items;
   drop those that no longer serve a target.
7. An idea that conflicts with a BRIEF Non-goal is dropped, not parked.

Gate sequence (moved from the BRIEF): Phase 1 profiling harness; Phase 2 Stage 1 framework
shootout (gate B1); Phase 3 traffic environment, built with the shootout winner and
auto-resetting; Phase 4 end-to-end GPU RL (the Stage 2 gate). The Stage 2 gate: a speedup over
the Stage 2 CPU reference with C1 to C3 passing, and learning that beats a random policy, then a
fixed-timer traffic light (the Stage 2 P1 heuristic); DECISIONS records the best measured
env-steps/hour against the aspirational 10^9 (BRIEF Targets).

Sweeps (owner, 2026-11-14; throughput is not monotonic, e.g. a smaller batch can beat a larger
one once state spills out of cache). Required: batch size, and steps fused per launch (it sets
C2's threshold). Once per framework: telemetry interval, memory layout, integer width, and
threads per block (custom kernels only). Run each on the GTX 1070 and record it in DECISIONS.md,
Sweep results. Re-run a sweep only when the framework or memory layout changes.

## Gate B1: Stage 1 ≥ 10^6 gen/hour (P = 10,000, one population), correct and learning, with C1 to C3 passing

Done when: `uv run gpu-sim bench -- <stage-1 workload>` exits 0 on the GTX 1070 at P = 10,000
with a median of 3 runs ≥ 10^6 generations/hour over a 10^5–10^6-generation run; that workload matches the CPU reference (B12)
and beats the no-selection control (B13); DECISIONS.md records the shootout winner with that
evidence and its speedup over the CPU reference. The winner is the fastest candidate that
matches the reference and beats the control (rejected candidates need not finish); and the
owner has re-reviewed the BRIEF, after which the Phase 3 gate is written here. The batch-size
sweep (B10) is informational for Stage 1 and sets the town count N for Stage 2.

| ID | Item | Serves | Done when | Blocked by | Size |
|---|---|---|---|---|---|
| B15 | Investigate intermittent probe-test compile failure: a reviewer saw nvcc 12.9 fail on `tests/fixtures/probe.cu` with glibc 2.43 `sinpif`/`rsqrtf` noexcept mismatches, while the same test passes in the main session. The header patch from `~/.home/setup/cuda.sh` is applied (sha 16b0d2fa…); check whether a different CUDA_HOME/include path or env causes it | unblocks B1 | the cause is found and the test passes in a fresh session, or the required env is documented in CLAUDE.md | | S |
| B2 | Trivial fused workload (Phase 1 done-criterion; the probe fails C1/C2 by design) | unblocks B5–B9 | `gpu-sim bench -- <trivial workload>` exits 0 on the GTX 1070, and a `@pytest.mark.gpu` test runs it | | S |
| B12 | Stage 1 CPU reference: a small NumPy implementation of beak vs flower co-evolution with a trait-match metric. It is the correctness oracle, the speed baseline and the spec B7–B9 implement | unblocks B7–B9, B13 | a pytest runs it on fixed seeds; DECISIONS Current measurements records its gen/hour (median of 3) at P = 10,000 | | M |
| B13 | Stage 1 no-selection control: trait-match metric over generations, selection vs no selection, on the CPU reference | P1 validity (Stage 1 learning); unblocks B1 | DECISIONS records both curves on ≥ 3 fixed seeds; selection beats the control by more than the spread across seeds | B12 | S |
| B3 | Count `cuMemcpy2D*` / `cuMemcpy3D*` (and their async/peer variants) as host copies when host-side | C1 (invariant) | a unit test classifies those names; DECISIONS T2 updated | | S |
| B4 | JAX `host_syncs` gap: XLA waits through `cuEventRecord` / `cuLaunchHostFunc` / `cuStreamWaitEvent`, not `cu*Synchronize`, so `host_syncs` reads 0 (D2H readbacks are still counted) | C1 (invariant); unblocks B7 | bench counts JAX host waits, or DECISIONS records why the D2H count suffices, with a test either way | | M |
| B5 | Steps-fused-per-launch sweep: steps per launch vs steps/hour, also compile time and VRAM; re-run per framework | C2 (invariant) | DECISIONS sweep table + takeaway; C2's (measured) threshold confirmed or changed in both BRIEF C2 and `bench.LAUNCHES_PER_10K_MAX` (BRIEF P2) | B2 | M |
| B6 | Counter overhead: the same workload's steps/hour with and without `gpu-sim bench` (expected ≈ 0) | Stage 1 throughput (req 2) | DECISIONS records both medians of 3; the difference is inside the noise band, or the bias is noted | B2 | S |
| B7 | Stage 1 in JAX: vectorised beak vs flower co-evolution, P = 10,000, 10^5–10^6 generations back to back in VRAM. Measure `launches_per_10k` early (an XLA loop may launch per iteration unless command buffers are on) | Stage 1 throughput (req 2) | a pytest matches the B12 reference on fixed seeds within a stated tolerance and reproduces B13's selection-beats-control result; bench median of 3 and speedup over B12 recorded in DECISIONS (T10 evidence); or dropped to Rejected with failed SM 6.1 check | B2, B4, B12 | M |
| B8 | Stage 1 in PyTorch with CUDA Graphs (torch pin T6) | Stage 1 throughput (req 2) | as B7 | B2, B12 | M |
| B9 | Stage 1 in one custom-kernel backend (Warp/Taichi/Triton/CUDA); no SM 6.1 build means drop it; Warp, Taichi and Triton are uv dependencies, not new toolchains; only raw CUDA C++ (a new language) needs the owner first (BRIEF P4) | Stage 1 throughput (req 2) | as B7 | B2, B12 | L |
| B10 | Batch-size sweep (population P) per candidate: find the peak and where it falls off; check C3 at the peak. Informational for Stage 1 (the gate stays at P = 10,000); it sets N for Stage 2 | Stage 1 throughput (req 2) | DECISIONS sweep table + takeaway per candidate | B7–B9 | M |
| B11 | Once-per-framework sweeps for each surviving candidate: telemetry interval (10^5 to 10^6; C1 forbids shorter), memory layout AoS vs SoA (often 2–5× on Pascal), integer width int32/int16/int8 (narrower saves bandwidth; some ops are slower), threads per block (custom kernels only) | Stage 1 throughput (req 2) | DECISIONS has each sweep's table + takeaway | B7–B9 | M |

## Parked

| ID | Item | Serves | Done when |
|---|---|---|---|
| B16 | Showcase visuals: one generic, small way to render both stages with no overhead on benchmarked runs (idea: re-run a short seeded slice outside the bench, emit fixed-shape 2D point frames, animate with matplotlib in `scripts/`) | Goal (showcase); BRIEF Non-goals | a design is agreed with the owner, then one Stage 1 animation is produced by a committed script |
