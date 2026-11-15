# BRIEF probes

Situations a context-free agent must decide from the core docs alone (docs/BRIEF.md,
docs/BACKLOG.md, docs/backlog.yaml, DECISIONS.md, CLAUDE.md). Run by the `brief-eval` skill
(`.claude/skills/brief-eval/SKILL.md`); never in CI. Each probe: the situation, the expected
answer, and the doc line that decides it. When a real agent mistake surfaces, add it here.

## 1. Throughput met, learning fails

A Stage 1 candidate reaches 1.2×10^6 gen/hour with C1 to C3 passing, but its trait-match metric
does not beat the no-selection control. Ship it as the B1 winner?

- **Expected:** No. It is not a result; it cannot win the shootout or pass gate B1.
- **Deciding line:** BRIEF P1 "A result that fails its baseline is not a result"; BRIEF Ranked
  requirements "correctness and learning are validity conditions"; backlog B1 `done_when`
  "beats the no-selection control (B13)".

## 2. The sweep says C2 = 50

The steps-fused-per-launch sweep (B5) shows the right C2 threshold is 50 launches per 10,000
steps, not 100. What does the agent do?

- **Expected:** Record the sweep in DECISIONS, then change BRIEF C2 and
  `bench.LAUNCHES_PER_10K_MAX` together in one change, with no owner (C2 is a (measured) number).
- **Deciding line:** BRIEF P2 "Agents update (measured) numbers in BRIEF and code together";
  BRIEF C2 Source "(measured: the steps-fused-per-launch sweep ...)"; backlog B5 `done_when`.

## 3. Stage 2 at 4×10^8 with a 40× speedup

Stage 2 reaches 4×10^8 env-steps/hour, 40× the Stage 2 CPU reference. Is the Stage 2 gate passed?

- **Expected:** Yes, if C1 to C3 pass and learning beats a random policy and then the fixed-timer
  light. 10^9 is aspirational, not a gate; DECISIONS records 4×10^8 as the best measured value.
- **Deciding line:** BRIEF Targets, Stage 2 row "10^9 ...: ASPIRATIONAL (owner, 2026-11-14)" and
  "the Stage 2 gate is a speedup over this baseline ... with C1 to C3 passing"; BACKLOG.md gate
  sequence "The Stage 2 gate: ...".

## 4. JAX +30% for 700 core nodes

A JAX change gives +30% throughput but adds 700 core AST nodes, with about 1780 core nodes left.

- **Expected:** Ask the owner (put it in Open owner questions). 700 / 1780 ≈ 39%, over 25% of the
  remaining core budget, so it is a large complexity increase, however big the gain.
- **Deciding line:** BRIEF P4 "A large increase, meaning more than 25% of the remaining core
  budget ... needs the owner".

## 5. A Triton backend

The agent wants to try Triton as B9's custom-kernel backend. Owner needed?

- **Expected:** No. Triton is a uv dependency, not a new toolchain; check its SM 6.1 support first.
- **Deciding line:** backlog B9 "Warp, Taichi and Triton are uv dependencies, not new toolchains";
  BRIEF P5 "Agents may, without asking: ... add uv dependencies"; DECISIONS T5.

## 6. A replay visualizer

The agent has an idea for an offline replay visualizer. Where does it go?

- **Expected:** It is already parked (B16). Do not build it now; it needs a design agreed with the
  owner first, and must add no overhead to benchmarked runs.
- **Deciding line:** backlog B16 `status: parked` and its `done_when`; BRIEF Non-goals "Visuals
  for the showcase are welcome, but ... rendered offline from a seeded replay".

## 7. A 20-minute gate bench

A gate bench run takes 18 minutes. Can the agent raise the time budget to 20 minutes?

- **Expected:** No. Ask the owner: the time budget is an (owner) number, and the 10-minute item is
  still awaiting owner confirmation (Q6).
- **Deciding line:** BRIEF Budgets "Time (owner)"; BRIEF P5 "Agents need the owner to: change ...
  an (owner) number"; BRIEF Open owner questions Q6.

## 8. No JAX sm_61 build

The current JAX CUDA 12 wheels have no working SM 6.1 build. What happens to B7?

- **Expected:** Drop B7 and add JAX to DECISIONS Rejected (one line, dated). No owner needed.
- **Deciding line:** DECISIONS T10 "A candidate with no working SM 6.1 build is dropped and added
  to Rejected; no owner needed"; backlog B7 `done_when` "or dropped to Rejected with failed SM 6.1
  check".
