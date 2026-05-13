# RFC: Auto-CUBE — Use-Case-Driven Judge + Coding-Agent Drivers

**Status:** DRAFT (revision 3 — concise rewrite, design decisions only)
**Author:** Alexandre Lacoste
**Date:** 2026-05-13

**Companions:** `trajectory-judge` (PR #366, the judge this builds on), `agent-owns-loop` (PR #386, orthogonal — touches the episode loop, not the judge).

---

## Problem

The trajectory judge (PR #366) is monolithic: one hard-coded prompt, one model, one tool surface, one transport (`claude-agent-sdk`). The meta-agent is a slash command that drives experiments and writes journals but has no typed contract with the judge. There is no story for:

- Running the judge **per use case** (general blame vs profiling vs scaffolding analysis) without forking the judge module.
- Running the judge **without an API key** (researcher with only a Claude Code subscription) or on top of a different coding agent (Codex).
- Letting the meta-agent **inject experiment-specific context** the judge should focus on.
- A **self-quality signal** — does the judge agree with itself? did it miss evidence?
- A **cross-experiment view** the outer loop can grep.

Auto-CUBE adds those four seams. It is post-hoc only — episode loop and storage formats are untouched.

---

## Design — five decisions

### 1. Each use case is a directory

`cube_harness/analyze/judge/use_cases/<name>/` holds the `JudgeRecipe` (Pydantic), the meta-agent skill (`SKILL.md`), the prompts, and any supporting scripts for that use case. `RECIPE_CATALOG` is assembled by walking the directory on import. New use cases are one-directory PRs.

Initial catalog: **`general_blame`**, **`profiling`**, **`agent_scaffolding`**. Others (`hint_harvest`, `auto_verified`) are deferred to follow-up PRs.

### 2. `AgentDriver` is our own Protocol — not LiteLLM

LiteLLM is a model gateway, not a coding-agent abstraction. To swap Claude Code SDK ↔ terminal `claude -p` ↔ `codex exec`, we need our own thin Protocol:

```python
class AgentDriver(Protocol):
    name: str
    max_parallelism: int
    async def run(...) -> DriverResult
    async def continue_session(...) -> DriverResult   # for audit pass
```

Three drivers in scope (Codex is signposted, not built):

| Driver | Auth | Parallelism per host | Cost reported |
|---|---|---|---|
| `claude-code-sdk` | API key | ~8 | yes |
| `claude-terminal` | subscription (`claude -p`) | ~2 | no |
| `codex-cli` (follow-up) | OpenAI key | ~3 | yes |

**Drivers are call-time, not recipe-pinned.** A researcher swaps `--driver claude-terminal` without touching the recipe. Underneath, model calls still route through LiteLLM when `LITELLM_PROXY_URL` is set, but that's per-driver implementation detail, not part of the Protocol.

### 3. Judge requires a context file — no fallback

The meta-agent writes `<experiment_dir>/judge_context.md` with the paths and prose the judge needs. `validate_context_file` is already seamed; the judge calls it and raises if the file is missing or any path doesn't resolve. The previous `collect_source_paths` venv-walk heuristic is removed from the judge's hot path.

For ad-hoc developer use, a new `ch-judge init-context <experiment_dir>` subcommand bootstraps the file using the old heuristic, then exits.

### 4. Audit pass — flag-gated, replaces `self_judge`

When `--audit` is set, the judge issues a follow-up turn (via `driver.continue_session`) asking it to critique its own reasoning. Output: `audit.json` next to `judge_output.json`. Off by default; ~25% cost overhead when on.

This is a flag, not a recipe — it always layers on top of an existing use case. A follow-up PR adds a batch step that walks a sweep, collects all `audit.json`s, re-judges flagged trajectories, and writes a sweep-level quality report.

### 5. Cross-judge agreement — same recipe, varying seeds

Agreement is `(recipe, trajectory)` × N seeds. Cross-recipe comparison is explicitly out — comparing `general_blame` to `agent_scaffolding` is only well-defined on shared core fields, and the more useful signal is "does this recipe reach the same blame twice?"

Output schema (`cross_judge_agreement.csv`) is fixed in this PR; the re-judging runner is a follow-up gated on low primary-blame confidence.

---

## Outputs

Per episode (inside `<episode_dir>/`):
- `judge_output.json` — from PR #366, unchanged
- `judge_trace.json` — from PR #366, unchanged
- `post_judge_survey.json` — second-pass self-assessment (ease, context quality, tooling gaps)
- `audit.json` — when `--audit` is set

Per experiment (`<experiment_dir>/`):
- `judge_context.md` — required input, written by meta-agent or `ch-judge init-context`
- `experiment_judge_report.csv` — from PR #366
- `cross_judge_agreement.csv` — when multiple seeds were judged

Per sweep (follow-up runner):
- `joint_judge_report.csv` — one row per (experiment, episode)
- `audit_quality_report.md` — follow-up batch-audit output

---

## Scope

**In (this PR):** `AgentDriver` Protocol + two drivers (`claude-code-sdk`, `claude-terminal`); `JudgeRecipe` widening (no `driver`/`selector` fields); the three initial use-case directories; required context file + `init-context` subcommand; survey collection; flag-gated audit pass; selector Protocol with three built-in selectors; fixed schemas for `joint_judge_report.csv` and `cross_judge_agreement.csv` (writers not yet implemented).

**Out (called-out follow-ups):** cross-judge runner; joint-CSV runner; batch-audit runner; Codex driver; real similarity selector; additional use-case directories; typed meta-agent loop (slash command stays).

**Not touched:** cube-standard, episode loop, trajectory format, storage protocol, `Episode`/`Experiment`/`exp_runner`, `LLM` wrapper.

---

## Constitution alignment

- **PS-002 (LiteLLM).** Drivers honour `LITELLM_PROXY_URL` and export `ANTHROPIC_BASE_URL` to their SDK/CLI when set. The constitution rule is satisfied at the token-routing layer; the agent-runtime abstraction is ours because LiteLLM does not cover it.
- **EX-002 (no global state).** No driver or recipe registry — `RECIPE_CATALOG` is assembled on import from the directory layout; drivers are passed as arguments.
- **CC-005 (minimalism).** Three use cases, two drivers, one flag for audit. Larger catalogs and runners come as separate PRs that prove their value.
- **SR-003 (escape hatch).** Both drivers expose `.raw` on `DriverResult`.
- **SR-002 (trace-first).** Driver calls are wrapped in OTel spans (`auto_cube.judge.driver_run`) with `auto_cube.driver`, `auto_cube.recipe`, `auto_cube.model`, `auto_cube.litellm_proxy` attributes.

---

## Open questions (for reviewer decision)

1. **`SKILL.md` registration.** Source-of-truth lives at `use_cases/<name>/SKILL.md`. Symlink them into `.claude/skills/` (lightweight, one source) or copy at build time (no symlink dep)? Recommend symlink unless someone objects.
2. **Driver Protocol parameter for the survey/audit pass session continuity.** `continue_session` on the SDK driver needs to span the survey-pass call in between the primary judgment and the audit. To verify against the SDK before merging — if continuation across intermediate calls doesn't work cleanly, the audit falls back to a fresh `run` with the prior judgment in the prompt. Either way, the surface area is the same.

Everything else (exact field names, audit prompt content, CLI flag spellings, OTel attribute keys, schema-version mechanics, default thresholds) is implementer's call — flag in PR review if you disagree.

---

## References

- `openspec/changes/trajectory-judge/proposal.md` — PR #366, the judge this extends.
- `openspec/changes/agent-owns-loop/proposal.md` — PR #386, orthogonal.
- `.claude/commands/meta-agent.md` — the slash command that drives the outer loop today.
- `src/cube_harness/analyze/judge/` — the judge package on `feat/trajectory-judge` where the seam PR landed `JudgeRecipe`, `PostJudgeSurvey`, `validate_context_file`, and the `related_trajectories` parameter this RFC builds on.
- LiteLLM Agent SDKs: https://docs.litellm.ai/docs/agent_sdks
- Claude Code headless: https://code.claude.com/docs/en/headless
- Codex non-interactive: https://developers.openai.com/codex/noninteractive
