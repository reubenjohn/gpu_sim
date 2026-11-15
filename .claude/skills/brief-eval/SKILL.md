---
name: brief-eval
description: Check that the core docs (BRIEF, BACKLOG, backlog.yaml, DECISIONS, CLAUDE.md) still lead a context-free agent to the right decisions. Use after any change to those files, or when an agent made a wrong call the docs should have prevented.
---

# brief-eval: probe the core docs with a context-free agent

The probes and their expected answers are in `evals/brief_probes.md`. This eval never runs in CI.

## When

- After any edit to `docs/BRIEF.md`, `docs/BACKLOG.md`, `docs/backlog.yaml`, `DECISIONS.md` or
  `CLAUDE.md`, before committing.
- When a real agent mistake surfaces: first add it as a new probe (situation, expected answer,
  deciding line), then run the eval.

## Steps

1. Read `evals/brief_probes.md`. Build a question list with only each probe's title and situation.
   Never pass the expected answers or deciding lines on.
2. Spawn one fresh, context-free subagent (Agent tool, `general-purpose`; not a fork, which would
   inherit your context). Its prompt:
   - Read only these files in the repo: `docs/BRIEF.md`, `docs/BACKLOG.md`, `docs/backlog.yaml`,
     `DECISIONS.md`, `CLAUDE.md`. No other files, no code, no git history.
   - For each situation, answer: what you do, whether the owner is needed, and the exact doc line
     that decides it (quoted). If the docs are silent or conflict, say so instead of guessing.
   - The question list from step 1.
3. Compare each answer with the expected answer:
   - **match:** same action and same owner/no-owner call;
   - **mismatch:** a different action or owner call;
   - **ambiguous:** right answer from a different or weaker line, or the agent flagged silence or a
     conflict.
4. Report a table: probe | verdict | agent's deciding line | the fix. For a mismatch or ambiguity,
   the fix is a doc edit (the smallest one that makes the deciding line unmissable) or, if the
   docs are right and the expectation is stale, a probe edit. Owner-gated text (BRIEF goal,
   principles, (owner) numbers) gets an Open owner question, not an edit.
5. If you edited docs, re-run the eval once with a new subagent.
