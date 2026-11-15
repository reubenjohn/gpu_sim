# Backlog

The items live in [backlog.yaml](backlog.yaml), in priority order: pick the first active item
whose `blocked_by` items are all gone. Its shape is enforced by
[backlog.schema.json](backlog.schema.json) (a pre-commit hook); caps and cross-references by
`tests/test_docs.py`. Only the orchestrating agent edits the backlog; subagents report items
upward.

Rules:
1. Every item names the [BRIEF](BRIEF.md) requirement or target it serves, or "unblocks B<n>"
   (it then inherits that item's priority). An item serving nothing is dropped.
2. Order: blocking items first, then the BRIEF rank of what it serves, then the cheapest first.
   Agents re-sort without asking.
3. Each item has a checkable `done_when`.
4. Finished items are DELETED, not checked off: git history keeps them and
   [DECISIONS.md](../DECISIONS.md) keeps results and evidence.
5. Only the NEXT milestone is a gate item (its `item` starts with "Gate"), with its work items
   after it. Later phases are not written out until they are next; their gate criteria apply the
   BRIEF constraints, end targets and P1 Baseline (each claim against its baseline). Every gate
   item's `done_when` includes the owner re-reviewing the BRIEF.
6. At most 15 `active` items; overflow becomes `parked`. Cap: 10 parked items; drop those that
   no longer serve a target.
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
