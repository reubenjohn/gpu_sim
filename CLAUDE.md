# gpu_sim

A 100% GPU-resident simulation and RL engine for a local GTX 1070 (Pascal SM 6.1, 8 GB, WSL2).

## Read first

1. [docs/BRIEF.md](docs/BRIEF.md) is the north star: constraints, ranked requirements, targets,
   budgets, principles P1 to P5 (P5 is the autonomy boundary) and open owner questions. Decide from it. Put anything you
   cannot decide in its "Open owner questions" rather than guessing.
2. [docs/BACKLOG.md](docs/BACKLOG.md) (rules) and [docs/backlog.yaml](docs/backlog.yaml) (items): pick the top unblocked item.
3. [DECISIONS.md](DECISIONS.md): current technical decisions with their evidence, sweep results
   and rejected options.

Only the orchestrating agent edits the backlog; subagents report items upward.

## Commands (uv only: never pip or system python)

- `uv sync`, then once per clone `uv run prek install`: install deps and the pre-commit hooks. `uv add <pkg>` / `uv add --dev <pkg>`: add a dependency.
- `uv run pytest -q`: unit tests, including the code-volume gate. `@pytest.mark.gpu` tests skip without the GPU, nvcc or gcc.
- `uv run prek run --all-files`: every hook in `.pre-commit-config.yaml` (ruff lint and format, lychee
  offline links, backlog schema, doc size caps), as CI runs it. Ruff: line length 120, C901 max 10.
- After editing BRIEF, BACKLOG, backlog.yaml, DECISIONS or CLAUDE.md, run the `brief-eval` skill
  (`.claude/skills/brief-eval/SKILL.md`; probes in `evals/brief_probes.md`).
- `uv run python scripts/code_volume.py`: per-file AST-node table against the budgets.
- `uv run gpu-sim bench [--keep DIR] -- <workload cmd>`: count a workload's CUDA driver calls with CUPTI and gate
  it on C1 to C3. The workload contract (cudaProfilerStart/Stop around the timed loop, plus a
  `BENCH_RESULT {...}` stdout line) is in the docstring of `src/gpu_sim/bench.py`. It exits 1 on
  a constraint failure.

## Layout

- `src/gpu_sim/`: core package. It counts toward `CORE_BUDGET`. `bench.py` is the Phase 1 harness.
- `scripts/`: dev tooling. `tests/`: pytest. Both count toward `TOTAL_BUDGET` only.
- `.github/workflows/ci.yml`: CPU-only CI (prek hooks, pytest). GPU gates run locally only.

## Step-loop rules (hard constraints C1, C2, C5, C6; see BRIEF "Hard constraints")

1. **No dynamic shapes.** Never use `tensor[mask]`, growing lists, or host-side `if` on tensor
   values inside the step loop. Use fixed padded tensors and `where(cond, new, old)`.
2. **FP32 and integers only.** Never enable float16/bfloat16 or autocast (why: BRIEF C5, DECISIONS T12).
   Use fp32 for kinematics and weights, int32/int16/int8/uint8 for indices and discrete states.
3. **No host round-trips.** No host↔device copies or host syncs in the timed loop beyond one packed telemetry readback per
   10^5 steps. Fuse thousands of steps into one XLA loop, CUDA Graph or megakernel, because
   each kernel launch is expensive under WSL2.
4. **Log every pivot.** Before proposing a framework or layout change, record its `gpu-sim bench`
   numbers (median of 3) as the evidence for that decision in DECISIONS.md (BRIEF P1).
5. **Toolchain.** CUDA 13 dropped Pascal. Use CUDA 12.x framework wheels and confirm SM 6.1
   support before you adopt one (DECISIONS T5 to T7).

## Code volume (approximate complexity; do not game it)

`tests/test_code_volume.py` caps AST nodes: `CORE_BUDGET` for `src/gpu_sim/`, and a much
larger `TOTAL_BUDGET` for all project Python, tests included. Names, comments, docstrings, type
hints and imports are free. The count is only a proxy for complexity, so never golf it with
packed one-liners, cryptic names or dense tricks. The budget is backpressure, so when it binds:
- find a more elegant solution: simplify, or reuse a library or the tool's own reports instead
  of hand-rolling. For example, use CUPTI callbacks rather than hand-parsing profiler databases.
- or push back on the requirement driving it, at the start of a session (BRIEF P4, "Push back
  early"). Don't ask for more budget to fit an inelegant path. Never change it yourself.
The budget is what lets agents work without the owner: under it, follow the BRIEF autonomy boundary (P5).

## Conventions

- Verify against reality. A perf claim needs a `gpu-sim bench` run on the GTX 1070, and "CI is
  green" needs `gh run view` showing green.
- Promote any shell pipeline or inline script you would reuse to `scripts/` or a test.
- The other ~1 GB of VRAM belongs to WSL2/OS (C3: peak < 7.0 GB).
- Toolchain setup (CUDA toolkit) needs `sudo` and is done by the owner. Ask; never run it yourself.
