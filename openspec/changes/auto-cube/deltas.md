# Deltas: Auto-CUBE

Applies to: `openspec/specs/analyze/spec.md` (primary), `openspec/specs/storage/spec.md` (minor), `openspec/specs/experiment/spec.md` (minor, joint-CSV pointer).

The seam PR (forward-compat to PR #366) already landed `JudgeRecipe`,
`PostJudgeSurvey`, `validate_context_file`, and the `related_trajectories`
parameter as no-op extensions. These deltas widen those types into a
first-class surface and add the coding-agent driver Protocol, selector
Protocol, audit pass, cross-experiment artifacts, and the use-case
directory layout.

Revision 2 (2026-05-13) reflects design-review feedback:
- Driver moves from recipe field to call-time argument.
- Context file becomes required (no `collect_source_paths` fallback).
- `self_judge` recipe replaced with a flag-gated audit pass.
- Initial use-case catalog is three (not six).
- LiteLLM proxy integration documented per PS-002.
- Recipe directory layout becomes one-dir-per-use-case under `use_cases/`.

---

## ADDED — `cube_harness/analyze/judge/driver.py`

### Public types

- `TraceMode` — `Literal["actions", "full", "off"]` (moved from `sdk.py`; the
  literal is re-exported from `sdk.py` for backwards compatibility).
- `ToolAction` — `TypedBaseModel(tool: str, input_summary: str, raw_input: dict[str, Any] | None)`.
  Replaces today's untyped `dict[str, Any]` entries in the `actions` list returned by
  `_run_claude_code`. Pydantic serialisation maps cleanly into `judge_trace.json`.
- `DriverResult` — `TypedBaseModel(output_text, prompt_tokens, completion_tokens, cost_usd, duration_s, actions: list[ToolAction], litellm_proxy_url: str | None = None, session_id: str | None = None)`.
  The driver-shaped result; consumed by `_judge_episode_impl` to build `JudgeMetadata` and `judge_trace.json`.
  `litellm_proxy_url` is set when `LITELLM_PROXY_URL` was honoured. `session_id` is set when the driver supports `continue_session` (used by the audit pass).
- `AgentDriver` — `Protocol` with attributes `name: str`, `max_parallelism: int`, and methods:
  - `async def run(*, system_prompt, user_prompt, cwd, additional_dirs, model, allowed_tools, verbose=False, trace_mode="actions") -> DriverResult`.
  - `async def continue_session(*, follow_up_prompt, verbose=False, trace_mode="actions") -> DriverResult`. Drivers without continuation raise `NotImplementedError`; the audit pass falls back to a fresh `run` with the prior judgment serialised into the prompt.
- `ClaudeCodeSDKDriver` — concrete `AgentDriver` wrapping `claude-agent-sdk`.
  `name = "claude-code-sdk"`, `max_parallelism = 8`. Refactor of today's `_run_claude_code` into the Protocol shape; bit-identical observable behaviour when called from `_judge_episode_impl` for the `general_blame` recipe.
- `TerminalClaudeDriver` — concrete `AgentDriver` wrapping the `claude` CLI via `asyncio.subprocess`.
  `name = "claude-terminal"`, `max_parallelism = 2`. `cost_usd` and token counts are reported as `0`. JSON extraction reuses `analyze.judge.sdk._extract_json_block`. `continue_session` uses `claude --continue` against the most recent session ID.

### LiteLLM proxy routing (PS-002)

Both concrete drivers honour `LITELLM_PROXY_URL` and `LITELLM_PROXY_AUTH_TOKEN` env vars:

- When `LITELLM_PROXY_URL` is set, the driver exports `ANTHROPIC_BASE_URL=$LITELLM_PROXY_URL` and the matching auth header before invoking the SDK / CLI subprocess. The SDK / CLI then routes through the LiteLLM proxy, which itself routes to Anthropic / Bedrock / Vertex / Azure per the LiteLLM config.
- When unset, the driver falls back to the SDK / CLI default (direct Anthropic). API-key-based developer flows are unchanged.
- The driver records `litellm_proxy_url` (URL only, no credentials) on `DriverResult`. This propagates into `judge_metadata.json` and the joint CSV's `litellm_proxy_url` column.

### Invariants

- `AgentDriver.run` MUST be implemented as `async def`. Synchronous wrappers belong in callers.
- `AgentDriver.continue_session` MUST raise `NotImplementedError` when not supported (rather than returning a stub result).
- `max_parallelism` is advisory. `judge_experiment(n_parallel=...)` clamps against `driver.max_parallelism` and logs the clamp.
- `DriverResult.actions` MUST be empty when `trace_mode == "off"`.
- `DriverResult.cost_usd` of `0.0` is a documented sentinel for "driver does not report cost"; the joint CSV's `driver` column lets consumers scope summations to metered drivers.
- Drivers MUST NOT log credentials when `LITELLM_PROXY_AUTH_TOKEN` is in the environment.

---

## ADDED — `cube_harness/analyze/judge/use_cases/`

A package containing one subdirectory per judge use case. Each subdirectory contains the
recipe, a `SKILL.md` for the meta-agent, prompt templates, and any supporting scripts.

### Layout

```
use_cases/
├── __init__.py           # exports RECIPE_CATALOG; walks subdirectories on import
├── general_blame/
│   ├── __init__.py
│   ├── recipe.py         # exports RECIPE: JudgeRecipe
│   ├── SKILL.md          # meta-agent skill description
│   ├── prompts/
│   │   ├── system.md
│   │   ├── user.md
│   │   └── audit.md      # shared by all use cases via symlink in V1
│   └── scripts/          # optional helper scripts (empty here)
├── profiling/
│   ├── __init__.py
│   ├── recipe.py
│   ├── SKILL.md
│   ├── prompts/
│   │   ├── system.md
│   │   └── user.md
│   └── scripts/
│       └── summarise_profile.py
└── agent_scaffolding/
    ├── __init__.py
    ├── recipe.py
    ├── SKILL.md
    ├── prompts/
    │   ├── system.md
    │   └── user.md
    └── scripts/
```

### Public types

- `RECIPE_CATALOG: dict[str, JudgeRecipe]` — name → recipe. Imported and used by the CLI's `--recipe NAME` flag and by `judge_episode(..., recipe=...)`. Assembled by `__init__.py` walking subdirectories and importing each `recipe.py`'s `RECIPE` constant.

### Public recipes (Pydantic instances)

- `general_blame: JudgeRecipe` — current default. Identical observable behaviour to today.
- `profiling: JudgeRecipe` — narrower taxonomy, profiling-focused user-prompt template, `BashOutput` added to `allowed_tools`. Activated by the meta-agent when an experiment is flagged `profiling=True`.
- `agent_scaffolding: JudgeRecipe` — deeper agent-failure ontology. Output schema extends `JudgeOutput` with a sibling field `scaffold_diagnosis: ScaffoldDiagnosis | None` (default `None`; non-breaking for other recipes).

`hint_harvest`, `auto_verified`, and any other use cases are deferred to follow-up PRs (one directory per PR).

### `SKILL.md` registration

A small script (`scripts/sync_judge_skills.py`, committed in this PR) creates symlinks at `.claude/skills/judge-<name> -> src/cube_harness/analyze/judge/use_cases/<name>/SKILL.md`. The meta-agent reads the symlinked files; the source of truth is the use-case directory.

### Invariants

- Every recipe in `RECIPE_CATALOG` has a unique `name` field equal to its directory name.
- Each `use_cases/<name>/recipe.py` exports exactly one module-level `RECIPE: JudgeRecipe`.
- Each `use_cases/<name>/SKILL.md` exists and is non-empty.
- A recipe's `model` SHOULD be a string accepted by both `ClaudeCodeSDKDriver` and `TerminalClaudeDriver`.
- Recipes are pure data: importing the use-case module MUST NOT trigger network or filesystem I/O beyond what Pydantic does to validate field defaults.

---

## ADDED — `cube_harness/analyze/judge/audit.py`

The flag-gated audit pass.

### Public types

- `AuditOutput` — `TypedBaseModel(schema_version=1, recipe, driver, reasoning_quality: int [0..5], missed_evidence: list[str], alternative_blames: list[BlameAlternative], verdict: Literal["sound", "questionable", "wrong"], notes: str | None)`.
- `BlameAlternative` — `TypedBaseModel(blame: str, rationale: str)`.
- `AUDIT_FILENAME = "audit.json"`.

### Public functions

- `run_audit_pass(*, recipe: JudgeRecipe, driver: AgentDriver, judge_output: JudgeOutput, judge_trace_path: Path, survey_path: Path, audit_prompt: str) -> AuditOutput` — issues `driver.continue_session(follow_up_prompt=audit_prompt)` and parses the result. Falls back to `driver.run(...)` with the prior judgment serialised into the user prompt when `continue_session` raises `NotImplementedError`.

### Invariants

- `AuditOutput.schema_version` is fixed at `1` for this RFC; subsequent schema changes increment the integer in lockstep with `JUDGE_SCHEMA_VERSION` semantics.
- `AuditOutput.alternative_blames` is empty when `verdict == "sound"`.
- `audit.json` is written next to `judge_output.json` and only when the audit pass is enabled.

---

## ADDED — `cube_harness/analyze/judge/selection.py` (extensions)

The existing module gains a `Selector` Protocol and three built-in selectors. Existing public functions (`discover_episodes`, `select_episodes`, `_load_episode_record`, `EpisodeRef`) are unchanged.

### Public types

- `Selector` — `Protocol` with `name: str`, `k: int`, and `def select(*, main_episode: EpisodeRef, experiment_dir: Path, all_refs: Sequence[EpisodeRef]) -> list[Path]`.
- `SameTaskDifferentAgent` — `Selector` implementation; returns up to `k` episode paths from sibling experiments that share the main episode's `task_id` but differ on `agent_config._type`.
- `SameAgentPreviousIteration` — `Selector` implementation; returns up to `k` predecessor episodes from the same `family_id`.
- `TopKBySimilarityStub` — `Selector` implementation; returns `k` most-recent neighbours by directory mtime. Placeholder for a real scorer.

### Invariants

- `Selector.select` MUST NOT return paths that resolve to `main_episode.episode_dir`.
- Returned paths MUST exist on disk (selectors filter their candidates accordingly).
- The list length is bounded by `k`; selectors return fewer when fewer candidates qualify.

---

## ADDED — `cube_harness/analyze/cross_experiment/joint_csv.py`

Schema-only in the first PR; the runner that walks a sweep directory and writes the file ships in a follow-up.

### Public types

- `JOINT_REPORT_FILENAME = "joint_judge_report.csv"`
- `JOINT_REPORT_COLUMNS: tuple[str, ...]` — fixed column order, listed in the proposal. Includes `litellm_proxy_url` and `driver` columns alongside the per-experiment columns from PR #366.

### Public functions

- `write_joint_csv(sweep_dir: Path, *, experiment_dirs: Sequence[Path] | None = None) -> Path` — declared in this PR with a `raise NotImplementedError("ships in follow-up PR")` body so callers and tests can be wired up. The schema is normative.

### Invariants

- Exactly one row per `(experiment_id, trajectory_id)`.
- All columns in `JOINT_REPORT_COLUMNS` are present even when empty (empty string for missing cross-judge columns).
- The file is written atomically (write to `.tmp`, rename).

---

## ADDED — `cube_harness/analyze/cross_experiment/cross_judge_agreement.py`

Schema-only in the first PR; the runner ships in a follow-up. **Single recipe, multiple seeds** (revision 2): agreement is per `(recipe, trajectory)` pair across seeds, not across recipes.

### Public types

- `AGREEMENT_REPORT_FILENAME = "cross_judge_agreement.csv"`
- `AGREEMENT_COLUMNS: tuple[str, ...]` — fixed column order: `trajectory_id`, `recipe`, `n_judgments`, `seeds`, `primary_blame_modal`, `primary_blame_agreement`, `outcome_modal`, `outcome_agreement`, `confidence_mean`.

### Public functions

- `write_cross_judge_agreement(experiment_dir: Path, *, judgments: dict[tuple[str, str], list[tuple[JudgeOutput, JudgeMetadata]]]) -> Path` — declared with `NotImplementedError` body. The dict is keyed by `(trajectory_id, recipe_name)`; each value is the list of seed-varying judgments for that pair. Schema is normative.

### Invariants

- One row per `(trajectory_id, recipe)` pair with `n_judgments >= 2`. Single-judgment pairs do not appear.
- `primary_blame_agreement` and `outcome_agreement` are in `[0.0, 1.0]`.
- `seeds` column is semicolon-separated.
- The runner only re-judges with the **same recipe** at different seeds; cross-recipe agreement is explicitly out of scope for this artifact.

---

## MODIFIED — `cube_harness/analyze/judge/recipe.py`

The seam PR landed `JudgeRecipe(name, system_prompt, model, allowed_tools)`. Auto-CUBE widens the fields and tightens types.

### Public types

- `JudgeRecipe` gains fields:
  - `user_prompt_template: str` — `.format()`-style template; the same kwargs that `analyze.judge.prompt.build_user_prompt` accepts.
  - `audit: bool` — default `False`. Gates the audit pass (which can also be enabled at call time via `--audit`).
  - `post_judge_survey: bool` — default `True`. When `False`, the second-pass survey is skipped and `post_judge_survey.json` is written with `recipe=<name>`, `schema_version=1`, all other fields `None`/empty.
  - `permission_mode: Literal["bypassPermissions", "ask"]` — default `"bypassPermissions"`.
  - `allowed_tools: tuple[str, ...]` — change from `list[str] | None` to `tuple[str, ...]`. Default `("Read", "Glob", "Grep", "Bash")`. The `None` shape from the seam PR is no longer valid (was a placeholder while the field was unplumbed); migration is mechanical because no production caller passes `None` today.

### NOT added (revision 2)

- **`driver` field is NOT added to `JudgeRecipe`.** Drivers are call-time arguments, not recipe fields. This kills the need for a `DRIVER_REGISTRY` (recipes never reference drivers, so JSON serialisation is straightforward Pydantic; no custom serialiser).
- **`selector` field is NOT added to `JudgeRecipe`.** Selectors are call-time arguments, mirroring driver placement.

### Invariants

- `JudgeRecipe.allowed_tools` is non-empty.
- `JudgeRecipe` serialises to JSON without custom serialisers.

### Public constants

- `DEFAULT_RECIPE` — unchanged identity (still the `general_blame` recipe), but now imported from `analyze/judge/use_cases/general_blame/recipe.py` rather than defined inline. `analyze/judge/recipe.py` re-exports it.

---

## MODIFIED — `cube_harness/analyze/judge/core.py`

### `judge_episode` signature

Adds `recipe: JudgeRecipe | None = None`, `driver: AgentDriver | None = None`, `selector: Selector | None = None`, `audit: bool = False`. When `recipe is None`, `DEFAULT_RECIPE` is used; when `driver is None`, `ClaudeCodeSDKDriver()` is constructed. `audit=True` enables the audit pass for this call (overrides `recipe.audit=False`).

```python
def judge_episode(
    episode_dir: Path,
    *,
    experiment_dir: Path | None = None,
    recipe: JudgeRecipe | None = None,
    driver: AgentDriver | None = None,
    selector: Selector | None = None,
    audit: bool = False,
    verbose: bool = False,
    trace_mode: TraceMode = "actions",
) -> tuple[JudgeOutput, JudgeMetadata]: ...
```

The `model: str = DEFAULT_MODEL` kwarg is **removed**: model is part of the recipe. A deprecation note in the function docstring tells callers to migrate to `recipe=`.

The `context_file: Path | None = None` parameter from revision 1 is **removed**: the context file is no longer optional. Its location is derived from `experiment_dir` (or from the `episode_dir`'s parent when `experiment_dir` is `None`) by `find_default_context_file`. If the file is missing, `_judge_episode_impl` raises `FileNotFoundError`.

### `judge_experiment` signature

Adds `recipe: JudgeRecipe | None = None`, `driver: AgentDriver | None = None`, `selector: Selector | None = None`, `audit: bool = False`. Drops `model: str`.

```python
def judge_experiment(
    experiment_dir: Path,
    *,
    recipe: JudgeRecipe | None = None,
    driver: AgentDriver | None = None,
    selector: Selector | None = None,
    audit: bool = False,
    ids: list[str] | None = None,
    sample: float | None = None,
    n: int | None = None,
    failures_only: bool = False,
    overwrite: bool = False,
    seed: int | None = None,
    verbose: bool = False,
    n_parallel: int = 1,
    trace_mode: TraceMode = "actions",
) -> dict[str, tuple[JudgeOutput, JudgeMetadata]]: ...
```

`n_parallel` is silently clamped to `driver.max_parallelism` with a single info-level log line.

### `_judge_episode_impl` (internal)

- Accepts `recipe: JudgeRecipe`, `driver: AgentDriver`, `selector: Selector | None`, and `audit: bool`.
- Calls `find_default_context_file(experiment_dir)` and then `validate_context_file(...)`. Raises `FileNotFoundError` if the file is missing or any referenced path does not resolve. **No fallback to `collect_source_paths`.**
- Calls `selector.select(...)` when `selector is not None`, and passes the returned related-trajectory paths to the user-prompt template (the template's `{related_trajectories}` placeholder is added; `general_blame` ignores it).
- Calls `driver.run(...)` instead of `_run_claude_code(...)` directly.
- After parsing `JudgeOutput`, when `recipe.post_judge_survey` is `True`, runs a second `driver.run(...)` pass with the survey prompt, parses the result against `PostJudgeSurvey`, and writes `post_judge_survey.json` next to `judge_trace.json`.
- When `audit or recipe.audit`, runs `audit.run_audit_pass(...)` and writes `audit.json` next to `judge_output.json`.

### Invariants (unchanged from PR #366)

- `_validate_invariants(judge_output)` runs unchanged.
- `_persist_judgment` writes `judge_output` and `judge_metadata` into `episode_record.json` or a sidecar.
- Aggregate cost in `experiment_judge_summary.json` excludes survey-pass and audit-pass cost by default; additive `total_survey_cost_usd` and `total_audit_cost_usd` fields are appended so consumers can opt in.

---

## MODIFIED — `cube_harness/analyze/judge/context.py`

`validate_context_file` is already present (seam PR). Auto-CUBE promotes it from "available when called" to "called by default and required". Helper additions:

- `find_default_context_file(experiment_dir: Path) -> Path` — returns `experiment_dir / "judge_context.md"`. Raises `FileNotFoundError` when the file is absent. Used by `_judge_episode_impl`.

### REMOVED — `collect_source_paths` from the judge's hot path

`collect_source_paths` continues to exist as the implementation of `ch-judge init-context` (it walks the venv to bootstrap a starter `judge_context.md`). It is **not** called by `_judge_episode_impl` any more. The judge's runtime path is exclusively through `validate_context_file`.

### Invariants

- `validate_context_file` raises `FileNotFoundError` on any missing referenced path. The judge MUST surface this — it is an outer-loop bug, not a judge-time fallback condition.
- `find_default_context_file` is the only place the filename `judge_context.md` is hardcoded.

---

## MODIFIED — `cube_harness/analyze/judge/sdk.py`

The free function `_run_claude_code` is **deleted** after its body is moved into `ClaudeCodeSDKDriver.run`. The module retains:

- `_extract_json_block` (still imported by `core.py` and reused by the terminal driver).
- `_summarise_tool_input` (used by both drivers to populate `ToolAction.input_summary`).
- `TraceMode`, `JUDGE_ALLOWED_TOOLS` re-exported for backwards compatibility.

### Invariants

- `_extract_json_block`'s public behaviour is unchanged.
- `_summarise_tool_input` is moved here from `sdk.py` (where it already lives) and made driver-agnostic.

---

## MODIFIED — `cube_harness/analyze/judge/cli.py`

The CLI gains four flags and one subcommand. Existing flags (`--summary`, `--overwrite`, `--ids`, `--sample`, `--n`, `--failures-only`, `--seed`, `--verbose`, `--n-parallel`, `--trace-mode`) are unchanged.

- `--recipe NAME` — name from `RECIPE_CATALOG`. Default `general_blame`.
- `--driver {claude-code-sdk,claude-terminal}` — driver override. Default `claude-code-sdk`. Always applied uniformly across the invocation.
- `--audit` — enable the audit pass for this invocation (overrides `recipe.audit`).
- `--no-survey` — disable the post-judge survey pass for this invocation (override of `recipe.post_judge_survey`).

The `--model` flag is **removed** (model is part of the recipe). A backwards-compatibility shim accepts `--model X` and resolves to a synthetic recipe `JudgeRecipe(name="cli-override", ...).model_copy(update={"model": X})` based on the selected recipe, with a deprecation warning to stderr.

### NEW subcommand — `ch-judge init-context <experiment_dir>`

Walks `experiment_config.json` exactly as `collect_source_paths` does and writes a starter `judge_context.md` containing the discovered `paths` fence (no prose). Exits 0 on success, non-zero if `experiment_config.json` is unreadable. Idempotent — re-running overwrites the file with a warning.

The `--context-file` flag from revision 1 is **removed**: the context file location is fixed at `<experiment_dir>/judge_context.md`. Out-of-tree paths are not supported in V1.

---

## MODIFIED — `openspec/specs/analyze/spec.md`

A new section "Judge subsystem" is added (companion to the trajectory-judge section that lands with PR #366). The section documents:

- The use-case directory layout under `analyze/judge/use_cases/<name>/` and the `RECIPE_CATALOG` assembly.
- `JudgeRecipe` as the unit of judge configuration; `driver` and `selector` are call-time arguments, not recipe fields.
- The `AgentDriver` Protocol, the two concrete drivers, the `continue_session` audit seam, and the cost/parallelism trade-offs.
- The LiteLLM proxy integration via `LITELLM_PROXY_URL` (PS-002).
- The context-file contract: `judge_context.md` is **required**; `validate_context_file` is the contract enforcer; `ch-judge init-context` is the bootstrap subcommand.
- The post-judge survey schema (with `schema_version`) and when it is populated.
- The audit pass (flag-gated), `audit.json` schema, and the pointer to the follow-up batch-audit runner.
- The CLI surface above.
- The joint-CSV and cross-judge-agreement output filenames and column lists (schema-normative even when the runners ship in follow-ups).

---

## MODIFIED — `openspec/specs/storage/spec.md`

One additive sentence under the experiment-root artifacts table:

> When the outer-loop meta-agent dispatches a sweep, the sweep directory may contain `joint_judge_report.csv` (one row per `(experiment, episode)`) and `cross_judge_agreement.csv` (one row per multi-seed (recipe, trajectory) pair). Both are read-mostly; their schemas are fixed in the Auto-CUBE proposal. Per-episode artifacts may include `audit.json` next to `judge_output.json` when the audit pass was enabled.

No change to `EpisodeRecord` (PR #366's `judge_output` / `judge_metadata` already cover the per-episode surface).

---

## NOT CHANGED

- `JudgeOutput` schema — taxonomy, evidence, confidence fields, invariants. Recipes that need richer output add sibling fields (e.g. `agent_scaffolding` adds `scaffold_diagnosis`); the core schema is frozen.
- `JudgeMetadata` schema — unchanged. The terminal driver reports `cost_usd=0.0` per the documented sentinel; no new fields beyond the additive `litellm_proxy_url` carried via `DriverResult`.
- `EpisodeRecord.judge_output` / `EpisodeRecord.judge_metadata` — same fields, same write path. The seam PR's invariants hold.
- `Storage` Protocol — Auto-CUBE writes sidecar files (`judge_context.md`, `post_judge_survey.json`, `judge_trace.json`, `audit.json`, `joint_judge_report.csv`, `cross_judge_agreement.csv`); none of them go through the Storage interface.
- `Episode`, `Experiment`, `exp_runner` — the episode loop and experiment dispatch are untouched. Auto-CUBE is post-hoc only.
- `Trajectory`, `TrajectoryStep`, `AgentOutput` — no structural changes.
- `LLM` (`cube_harness.llm`) — the judge calls drivers, not the LLM wrapper. The constitution's LiteLLM requirement (PS-002) is satisfied via the LiteLLM-proxy env-var path that both drivers honour.
- `meta_agent/recipes/` — experiment-level recipes stay in place. Only judge recipes are new.
- `.claude/commands/meta-agent.md` — slash command stays. A follow-up RFC may codify its responsibilities into typed Python under `analyze/cross_experiment/meta_agent/`.
- Any benchmark or agent contracts. Auto-CUBE does not touch cube-standard.
