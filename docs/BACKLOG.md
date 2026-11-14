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
   written out until they are next (sketches for them are in DECISIONS.md, Open design variables).
6. At most ~15 active items; overflow goes to "Parked" at the bottom.

Sweeps: throughput is not monotonic (a smaller batch can beat a larger one once state spills out
of cache), so measure, don't assume. Run each sweep on the GTX 1070 and record a short table plus
a one-line takeaway in DECISIONS.md, Sweep results. Re-run a sweep only when the framework or
memory layout changes.

## Gate B1: Stage 1 ≥ 10^6 gen/hour (P = 10,000, one population) with C1 to C3 passing

Done when: `uv run gpu-sim bench -- <stage-1 workload>` exits 0 on the GTX 1070 with a median of
3 runs ≥ 10^6 generations/hour, and DECISIONS.md records the shootout winner with that evidence.
Then the owner re-reviews the BRIEF and the Phase 3 gate is written here.

| ID | Item | Serves | Done when | Blocked by | Size |
|---|---|---|---|---|---|
| B2 | Trivial fused workload (Phase 1 done-criterion; the probe fails C1/C2 by design) | unblocks B5–B9 | `gpu-sim bench -- <trivial workload>` exits 0 on the GTX 1070, and a `@pytest.mark.gpu` test runs it | | S |
| B3 | Count `cuMemcpy2D*` / `cuMemcpy3D*` (and their async/peer variants) as host copies when host-side | C1 (req 1) | a unit test classifies those names; DECISIONS T2 updated | | S |
| B4 | JAX `host_syncs` gap: XLA waits through `cuEventRecord` / `cuLaunchHostFunc` / `cuStreamWaitEvent`, not `cu*Synchronize`, so `host_syncs` reads 0 (D2H readbacks are still counted) | C1 (req 1); unblocks B7 | bench counts JAX host waits, or DECISIONS records why the D2H count suffices, with a test either way | | M |
| B5 | Steps-fused-per-launch sweep: steps per launch vs steps/hour, also compile time and VRAM; re-run per framework | C2 (req 1) | DECISIONS sweep table + takeaway; C2's threshold confirmed or changed in both BRIEF (owner) and `bench.LAUNCHES_PER_10K_MAX` | B2 | M |
| B6 | Counter overhead: the same workload's steps/hour with and without `gpu-sim bench` (expected ≈ 0) | Stage 1 throughput (req 3) | DECISIONS records both medians of 3; the difference is inside the noise band, or the bias is noted | B2 | S |
| B7 | Stage 1 in JAX: vectorised beak vs flower co-evolution, P = 10,000, 10^5–10^6 generations back to back in VRAM. Measure `launches_per_10k` early (an XLA loop may launch per iteration unless command buffers are on) | Stage 1 throughput (req 3) | bench median of 3 recorded in DECISIONS (T10 evidence) | B2, B4 | M |
| B8 | Stage 1 in PyTorch with CUDA Graphs (torch pin T6) | Stage 1 throughput (req 3) | as B7 | B2 | M |
| B9 | Stage 1 in one custom-kernel backend (Warp/Taichi/Triton/CUDA); no SM 6.1 build means drop it and add a Rejected line | Stage 1 throughput (req 3) | as B7, or a Rejected line in DECISIONS | B2 | L |
| B10 | Batch-size sweep (population P) per candidate: find the peak and where it falls off; check C3 at the peak | Stage 1 throughput (req 3) | DECISIONS sweep table + takeaway per candidate | B7–B9 | M |
| B11 | Once-per-framework sweeps for each surviving candidate: telemetry interval (10^4 to 10^6; C1 caps it at ≥ 10^5), memory layout AoS vs SoA (often 2–5× on Pascal), integer width int32/int16/int8 (narrower saves bandwidth; some ops are slower), threads per block (custom kernels only) | Stage 1 throughput (req 3) | DECISIONS has each sweep's table + takeaway | B7–B9 | M |
| B12 | Stage 1 CPU reference: a small CPU (NumPy) implementation the GPU result must match | Stage 1 learning quality (req 5) | a pytest compares CPU and GPU runs on fixed seeds within a stated tolerance | B7 or B8 or B9 | M |
| B13 | Stage 1 no-selection control: trait-match metric over generations, selection vs no selection | Stage 1 learning quality (req 5) | DECISIONS records both curves on fixed seeds; selection improves trait matching over the control | B7 or B8 or B9 | S |
| B14 | Set the exact Stage 1 learning-quality margin from the first runs | Stage 1 learning quality (req 5) | the margin is proposed to the owner and written into the BRIEF target | B13 | S |

## Parked

None.
