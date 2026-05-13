# Deltas: Auto-CUBE

Applies to: `openspec/specs/analyze/spec.md` (primary), `openspec/specs/storage/spec.md` (minor).

The seam PR (forward-compat to PR #366) already landed `JudgeRecipe`, `PostJudgeSurvey`, `validate_context_file`, and the `related_trajectories` parameter as no-op extensions. These deltas widen those types into a first-class surface and add the coding-agent driver Protocol, audit pass, and cross-experiment artifacts.

---

## ADDED — `cube_harness/analyze/judge/driver.py`

`AgentDriver` Protocol with `run(...)` and `continue_session(...)` async methods returning a `DriverResult` (output text, token counts, cost, duration, observed tool actions, optional LiteLLM proxy URL, optional session id).

Two concrete drivers: `ClaudeCodeSDKDriver` (`max_parallelism=8`, wraps `claude-agent-sdk`) and `TerminalClaudeDriver` (`max_parallelism=2`, wraps `claude -p` via subprocess; reports `cost_usd=0`). Codex driver is a signposted follow-up.

Both drivers honour `LITELLM_PROXY_URL` / `LITELLM_PROXY_AUTH_TOKEN` by exporting `ANTHROPIC_BASE_URL` to the underlying SDK/CLI. When unset, they use the SDK/CLI default. The URL (no credentials) is recorded on `DriverResult` and propagates into `judge_metadata.json`.

### Invariants

- `AgentDriver.run` is `async`. `continue_session` raises `NotImplementedError` when unsupported; callers fall back to a fresh `run` with prior context serialised into the prompt.
- `max_parallelism` is advisory — `judge_experiment` clamps and logs.
- `DriverResult.cost_usd == 0.0` is a sentinel for "driver does not meter."

---

## ADDED — `cube_harness/analyze/judge/use_cases/`

Each subdirectory is a self-contained use case: `recipe.py` (exports `RECIPE: JudgeRecipe`), `SKILL.md` (meta-agent skill), `prompts/`, optional `scripts/`. `__init__.py` walks subdirectories on import and assembles `RECIPE_CATALOG: dict[str, JudgeRecipe]`.

Initial directories: `general_blame`, `profiling`, `agent_scaffolding`. Adding a use case is a single PR that drops a new directory; no central registration touched.

A small script (`scripts/sync_judge_skills.py`) symlinks each `SKILL.md` into `.claude/skills/judge-<name>` so the meta-agent reads a single source of truth.

### Invariants

- Recipe `name` equals directory name.
- Importing a use-case module performs no network or filesystem I/O beyond Pydantic field validation.
- Recipes serialise to JSON without custom serialisers (no driver/selector fields).

---

## ADDED — `cube_harness/analyze/judge/audit.py`

Flag-gated audit pass. `run_audit_pass(...)` calls `driver.continue_session` with an audit prompt; falls back to `driver.run` when continuation is unsupported. Writes `audit.json` next to `judge_output.json` with fields covering `verdict` (`sound`/`questionable`/`wrong`), `reasoning_quality`, `missed_evidence`, and any `alternative_blames`.

Schema versioned via a `schema_version: int` field on `AuditOutput`, mirroring the existing `JUDGE_SCHEMA_VERSION` convention.

---

## ADDED — `cube_harness/analyze/judge/selection.py` (Selector extensions)

`Selector` Protocol with `select(*, main_episode, experiment_dir, all_refs) -> list[Path]`. Three built-in selectors: `SameTaskDifferentAgent`, `SameAgentPreviousIteration`, `TopKBySimilarityStub`.

Selectors are call-time arguments to `judge_episode` / `judge_experiment`, not recipe fields. Existing `EpisodeRef` / `discover_episodes` / `select_episodes` helpers are unchanged.

### Invariants

- A selector never returns the main episode's own path.
- Returned paths exist on disk; list length is bounded by `k`.

---

## ADDED — `cube_harness/analyze/cross_experiment/`

Two schema-only files in this PR (runners ship as follow-ups):

- `joint_csv.py` — `JOINT_REPORT_FILENAME = "joint_judge_report.csv"`, fixed column list including `driver`, `recipe`, and `litellm_proxy_url`. `write_joint_csv(...)` declared with `NotImplementedError`.
- `cross_judge_agreement.py` — `AGREEMENT_REPORT_FILENAME = "cross_judge_agreement.csv"`, fixed column list keyed by `(trajectory_id, recipe)` with a `seeds` column. **Single recipe, varying seeds only** — cross-recipe comparison is out of scope for this artifact. `write_cross_judge_agreement(...)` declared with `NotImplementedError`.

### Invariants

- Joint CSV: one row per `(experiment_id, trajectory_id)`; atomic write.
- Agreement CSV: one row per `(trajectory_id, recipe)` with `n_judgments >= 2`; agreements in `[0, 1]`.

---

## MODIFIED — `cube_harness/analyze/judge/recipe.py`

`JudgeRecipe` gains `user_prompt_template: str`, `audit: bool = False`, `post_judge_survey: bool = True`, `permission_mode: Literal["bypassPermissions", "ask"]`. `allowed_tools` tightens from `list[str] | None` to `tuple[str, ...]`.

**Does NOT gain `driver` or `selector` fields** — both are call-time arguments. No `DRIVER_REGISTRY`.

`DEFAULT_RECIPE` is re-exported from `use_cases/general_blame/recipe.py`.

---

## MODIFIED — `cube_harness/analyze/judge/core.py`

`judge_episode` and `judge_experiment` gain `recipe`, `driver`, `selector`, `audit` keyword arguments; lose `model` (now part of the recipe). `n_parallel` is clamped to `driver.max_parallelism` with a log line.

`_judge_episode_impl`:
- Calls `find_default_context_file(experiment_dir)` and `validate_context_file(...)`. **Raises** `FileNotFoundError` if the context file is missing or any path doesn't resolve. No fallback to `collect_source_paths`.
- Calls `selector.select(...)` when a selector is provided.
- Calls `driver.run(...)` instead of `_run_claude_code(...)`.
- Runs the survey pass when `recipe.post_judge_survey` is True; writes `post_judge_survey.json`.
- Runs the audit pass when `audit or recipe.audit`; writes `audit.json`.

Aggregate cost in `experiment_judge_summary.json` excludes survey and audit costs by default; additive `total_survey_cost_usd` and `total_audit_cost_usd` fields are appended.

---

## MODIFIED — `cube_harness/analyze/judge/context.py`

`find_default_context_file(experiment_dir: Path) -> Path` returns `experiment_dir / "judge_context.md"` and raises if absent.

`collect_source_paths` continues to exist as the implementation of the new `ch-judge init-context` subcommand. It is **no longer called by the judge's runtime path**.

---

## MODIFIED — `cube_harness/analyze/judge/sdk.py`

`_run_claude_code` is deleted; its body moves into `ClaudeCodeSDKDriver.run`. The module retains `_extract_json_block`, `_summarise_tool_input`, and re-exports `TraceMode` / `JUDGE_ALLOWED_TOOLS` for backwards compatibility.

---

## MODIFIED — `cube_harness/analyze/judge/cli.py`

New flags: `--recipe NAME`, `--driver {claude-code-sdk,claude-terminal}`, `--audit`, `--no-survey`. The `--model` flag is removed; a backwards-compatibility shim accepts it with a deprecation warning.

New subcommand: `ch-judge init-context <experiment_dir>` walks `experiment_config.json` and writes a starter `judge_context.md` (paths fence only).

---

## MODIFIED — `openspec/specs/analyze/spec.md`

Adds a "Judge subsystem" section documenting: the use-case directory layout, `JudgeRecipe` (with driver/selector as call-time arguments), the `AgentDriver` Protocol and its two concrete drivers, LiteLLM proxy routing, the required-context-file contract, the post-judge survey, the audit pass, the CLI surface, and the cross-experiment artifact schemas.

---

## MODIFIED — `openspec/specs/storage/spec.md`

One additive sentence under the experiment-root artifacts table covering `joint_judge_report.csv`, `cross_judge_agreement.csv`, and the per-episode `audit.json` when audit is enabled. No change to `EpisodeRecord`.

---

## NOT CHANGED

`JudgeOutput`, `JudgeMetadata`, `EpisodeRecord`, `Storage` Protocol, `Episode`, `Experiment`, `exp_runner`, `Trajectory`, `LLM`, `meta_agent/recipes/`, `.claude/commands/meta-agent.md`, cube-standard.
