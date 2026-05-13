# RFC: Auto-CUBE — Use-Case-Driven Judge, Coding-Agent Drivers, and the Meta-Agent Outer Loop

**Status:** DRAFT (revision 2 — folds in design-review feedback from 2026-05-13)
**Author:** Alexandre Lacoste
**Reviewer:** TBD
**Date:** 2026-05-13

**Companion RFCs:**
- `openspec/changes/trajectory-judge/proposal.md` (PR #366) — the inner-loop judge this
  RFC builds on.
- `openspec/changes/agent-owns-loop/proposal.md` — orthogonal; touches the episode
  loop, not the post-hoc judge or the meta-agent.

---

## Problem

cube-harness today has three pieces of post-hoc analysis machinery that were
designed in isolation and do not compose:

1. The **trajectory judge** in `cube_harness/analyze/judge/` (landing as PR #366).
   It produces a structured `JudgeOutput` per episode by invoking Claude Code over
   the `claude-agent-sdk`. Its prompt, model, and tool surface are hard-coded
   constants in `analyze/judge/prompt.py` and `analyze/judge/sdk.py`. There is one
   judge, and changing what the judge looks at means editing the module.

2. The **meta-agent** in `meta_agent/`, plus a fleet of `feat/meta-agent-*`
   branches. Recipes (e.g. `meta_agent/recipes/workarena_l1_full.py`), hints
   (`meta_agent/hints/swebench-verified.json`), tooling and result journals live
   side-by-side, but the workflow they implement is held together by a slash
   command (`.claude/commands/meta-agent.md`) rather than by typed Python.

3. **New ideas** from the user about how these pieces should compose into a
   long-term, self-improving system: judge use-cases packaged as cohesive units
   (judge recipe + meta-agent skill + supporting scripts), coding-agent
   abstraction so the judge can run on a subscription instead of an API key,
   joint analysis across experiments, an audit pass that closes the loop back to
   the judge's own quality.

The fragmentation has five concrete costs that this RFC addresses:

1. **The pre-judge owns context that should be the meta-agent's job.**
   `collect_source_paths` in `analyze/judge/context.py` resolves paths by
   importing `_type` strings from `experiment_config.json` against the local
   venv. This is correct for ad-hoc judging on a developer laptop but cannot
   express the per-experiment shape the meta-agent needs ("for this experiment,
   focus on the profiling trace; for that one, contrast with these three other
   trajectories"). The judge re-derives context every run and there is no place
   for an outer loop to inject curated context.

2. **No coding-agent abstraction.** `_run_claude_code` in `analyze/judge/sdk.py`
   talks to `claude-agent-sdk` directly. Researchers who hold a Claude Code
   subscription but no API key cannot run the judge. The Codex CLI cannot be
   bolted on. Even within the SDK path, there is no way to swap models or tool
   surfaces — the constants `JUDGE_ALLOWED_TOOLS` and
   `permission_mode="bypassPermissions"` are inlined.

3. **No feedback loop from judge to tooling.** When the judge runs into a gap —
   a missing profiling trace, an unparseable observation, a source tree it
   cannot reach — that experience evaporates. There is no record of *why* a
   judgment was hard, and no signal that lets us improve the judge's toolchain
   over time.

4. **No joint view across experiments.** `judge_experiment` writes one CSV per
   experiment (`experiment_judge_report.csv`). The meta-agent's natural unit of
   work is a *sweep* of experiments — half a dozen agent configs against the
   same benchmark, or the same agent across benchmarks. There is no row-per
   (experiment, episode) table the outer loop can grep.

5. **No agreement signal.** A single judge call is one sample from a noisy
   distribution; we have no way to ask "would the same recipe at a different
   seed reach the same blame on this trajectory?" — which is exactly the
   question the meta-agent needs to answer before recommending an intervention.

Auto-CUBE is the umbrella that unifies these threads. It is not a rewrite. It
is a contract that says: each use case is a directory containing a judge recipe
plus the meta-agent skill that drives it; the judge runs against an abstract
coding-agent driver chosen at call time; the meta-agent owns experiment-level
context and post-hoc analysis; and a flag-gated audit pass closes the loop
back to the judge's quality.

---

## Scope

### In (first Auto-CUBE PR)

- A **use-case directory layout** under `cube_harness/analyze/judge/use_cases/`.
  Each subdirectory contains the judge `recipe.py`, a `SKILL.md` for the
  meta-agent, and any supporting scripts/configs that use case needs. Initial
  catalog: `general_blame`, `profiling`, `agent_scaffolding` (three; the rest
  are deferred — see "Out").
- `JudgeRecipe` becomes the unit of judge configuration: name, system prompt,
  user-prompt template, model, allowed tools. The seam PR (forward-compat to
  #366) lands the Pydantic model with `name/system_prompt/model/allowed_tools`;
  this RFC widens it with the user-prompt template field. **`driver` is NOT a
  recipe field** — drivers are chosen at call time (see Driver section).
- An `AgentDriver` Protocol that abstracts the coding agent the judge runs as.
  Two implementations: `ClaudeCodeSDKDriver` (today's `_run_claude_code`
  refactored to fit the Protocol) and `TerminalClaudeDriver` (subprocess
  wrapper around the `claude` CLI for users with a subscription but no API key).
  Both route through the LiteLLM proxy when one is configured (see
  "Constitution alignment: PS-002").
- A `Selector` interface for related-trajectory selection, with three built-in
  selectors: `SameTaskDifferentAgent`, `SameAgentPreviousIteration`, and
  `TopKBySimilarityStub` (a stub that returns N most recent neighbours; real
  similarity scoring is a follow-up).
- `validate_context_file` (already seamed in `analyze/judge/context.py`)
  becomes the **only** context source. The judge requires
  `<experiment_dir>/judge_context.md` to exist and to validate; otherwise it
  raises. The current `collect_source_paths` heuristic is removed from the
  judge's hot path. A new CLI subcommand `ch-judge init-context
  <experiment_dir>` bootstraps a context file from the venv, so ad-hoc users
  can produce the file once and then judge against it.
- `post_judge_survey.json` collection: the empty stub from the seam PR is
  populated by a deterministic second pass after each judgment, recording how
  hard the analysis was, whether the context file was useful, and what tooling
  gaps were hit. Schema fixed in this RFC, with a `schema_version` field that
  mirrors the existing `JUDGE_SCHEMA_VERSION` convention.
- An **audit pass** (flag-gated): when a recipe's `audit=True` is set or
  `--audit` is passed at call time, the judge runs a second pass that
  *continues the prior judgment's context* — same driver session, same
  conversation, with a follow-up prompt asking the judge to critique its own
  reasoning. Output written to `audit.json` next to `judge_output.json`.
- A `meta_agent/` repackaging that defines what stays as the outer-loop runner
  vs. what moves into `analyze/judge/use_cases/<name>/` and
  `analyze/cross_experiment/`. No behaviour change at the meta-agent loop
  level; this is a re-homing of the modules.

### Out (later PRs, called out explicitly)

- **Codex driver.** The Protocol is shaped to admit it; the implementation is
  follow-up work. Codex is structurally similar to the terminal Claude case
  (subprocess, no programmatic streaming API) but has its own auth and
  config surface. Like the Claude drivers, it would route through LiteLLM's
  Codex SDK adapter when one is configured.
- **Cross-judge agreement runner.** The output schema and the columns it adds
  to the joint CSV are specified here; the runner that re-judges N times per
  episode at different seeds is a follow-up. Single-judgment runs today still
  get a stable record that agreement can be appended to later.
- **Joint CSV across experiments runner.** The schema is fixed here; the
  outer-loop module that writes it lives in the follow-up PR that wires the
  meta-agent to multi-experiment dispatch.
- **Audit-batch analysis.** The per-episode audit pass ships in the first PR.
  The meta-agent skill + script that *collects* all audits across a sweep,
  re-judges the previous batch's trajectories, and produces a quality report
  ships in a follow-up. Schema for the per-episode `audit.json` is fixed here
  so the batch tool can be written without churn.
- **Additional use-case directories** beyond the initial three:
  `hint_harvest`, `auto_verified`. Each is a separate, small follow-up PR that
  drops a new directory under `use_cases/`. The plumbing change is in this RFC;
  catalog growth is incremental.
- **Full meta-agent rewrite.** Use-case skills relocate under their use-case
  dirs; the loop driver stays as the slash command at
  `.claude/commands/meta-agent.md` until the dispatch and analysis steps are
  codified into typed Python under `analyze/cross_experiment/`.
- **cube-standard contract changes.** None — this entire RFC is harness-side.
- **Changes to the episode loop, trajectory format, or storage protocol.**
  Out — see `agent-owns-loop` for the loop work and `atlas-eval-log` for
  storage.

---

## Design

### Architecture overview

Auto-CUBE has two nested loops. The boundary between them is the experiment.

**Inner loop — per experiment, 5 to 10 episodes typically.**

```
task → agent → trajectory → judge(recipe, driver) → judge_output.json
                                                  + judge_trace.json
                                                  + post_judge_survey.json
                                                  + audit.json (if --audit)
```

The judge is configured by a `JudgeRecipe` (system prompt, model, allowed
tools, user-prompt template). It receives a main trajectory directory, the
**required** `judge_context.md` produced by the meta-agent (or by `ch-judge
init-context`), and an optional list of related-trajectory paths produced by
a `Selector`. The driver is supplied at call time (default
`ClaudeCodeSDKDriver`). The judge emits a `JudgeOutput` (taxonomy and
evidence; schema fixed by PR #366), a `judge_trace.json` (the agent's tool
calls during the judgment), a `post_judge_survey.json` (the second-pass
self-assessment), and — when audit is enabled — an `audit.json` (the
self-critique pass).

**Outer loop — per sweep, dispatched by the meta-agent.**

```
meta-agent dispatch → N experiments → joint CSV + cross-judge agreement
   → meta-agent analysis → Experiment Judge Report → meta-agent intervention
   → (optional) audit-batch pass over the previous sweep
```

The meta-agent generates the context file for each experiment, dispatches the
runs, lets the inner loop write its per-episode outputs, then concatenates
them into a row-per (experiment, episode) joint CSV. Optional cross-judge
agreement is computed by re-running the inner loop with the **same recipe**
at different seeds and comparing `primary_blame` distributions. The
meta-agent reads the joint CSV, writes an Experiment Judge Report (a markdown
document with cited evidence), and proposes interventions: prompt tweaks,
scaffold edits, hint additions, or recipe changes that feed back into the
next sweep. A separate audit-batch step (follow-up PR) collects the
`audit.json` files across the previous sweep, re-judges flagged trajectories,
and reports on judge quality.

The line between the loops is a single contract: the inner loop trusts the
context file produced by the outer loop and validates that its paths resolve.
That is the entirety of the inter-loop dependency. The judge knows nothing
about sweeps; the meta-agent knows nothing about the judge's internal
toolchain.

### Use-case directory layout

Each judge use case is a self-contained directory under
`cube_harness/analyze/judge/use_cases/`. The contents are uniform so the
catalog can be discovered programmatically and so contributors have one
template to follow:

```
use_cases/
├── __init__.py            # exports RECIPE_CATALOG: dict[str, JudgeRecipe]
├── general_blame/
│   ├── __init__.py
│   ├── recipe.py          # JudgeRecipe instance: RECIPE
│   ├── SKILL.md           # meta-agent skill description (claude-md-improver style)
│   ├── prompts/
│   │   ├── system.md      # JUDGE_SYSTEM_PROMPT for this use case
│   │   └── user.md        # user-prompt template
│   └── scripts/           # supporting scripts (optional; empty for general_blame)
├── profiling/
│   ├── recipe.py
│   ├── SKILL.md
│   ├── prompts/
│   └── scripts/
│       └── summarise_profile.py   # py-spy / flameprof helper invoked by the judge
└── agent_scaffolding/
    ├── recipe.py
    ├── SKILL.md
    ├── prompts/
    └── scripts/
```

`use_cases/__init__.py` walks the subdirectories on import, imports each
`recipe.py` for its `RECIPE` constant, and assembles `RECIPE_CATALOG` keyed by
directory name. Adding a new use case is a single PR that drops a new
subdirectory; no central registration code is touched. The `SKILL.md` files
are the canonical location for the markdown the meta-agent reads when
deciding which use case to dispatch — symlinked into `.claude/commands/` (or
its sibling skill location) by a small script committed alongside this RFC,
so each `SKILL.md` has exactly one source of truth.

The use-case directory is the unit of cohesion: a contributor working on
`profiling` edits one folder for the recipe, the prompts, the helper scripts,
and the meta-agent's view of when to use it. The previous proposal split
recipes (Python), prompts (constants in `analyze/judge/prompt.py`), and
skills (`.claude/commands/`); the new layout collocates them.

### Recipe surface

A `JudgeRecipe` is a Pydantic `TypedBaseModel` (the seam PR lands the type
under `cube_harness/analyze/judge/recipe.py`; this RFC widens it). The
constitution forbids YAML configuration — recipes are Python, instantiated by
recipe authors in `use_cases/<name>/recipe.py` and aggregated into
`RECIPE_CATALOG`. There is no recipe loader; you import a `JudgeRecipe` and
pass it to `judge_episode(..., recipe=...)`.

```python
class JudgeRecipe(TypedBaseModel):
    name: str
    system_prompt: str
    user_prompt_template: str
    model: str = "claude-sonnet-4-6"
    allowed_tools: tuple[str, ...] = ("Read", "Glob", "Grep", "Bash")
    audit: bool = False           # gate for the per-episode audit pass
    post_judge_survey: bool = True
    permission_mode: Literal["bypassPermissions", "ask"] = "bypassPermissions"
```

`tuple[str, ...]` rather than `list[str]` for `allowed_tools` because the
recipe is frozen at instantiation and a tuple makes that obvious; Pydantic
serialises tuples and lists identically.

**Critical change from revision 1:** there is no `driver` field on
`JudgeRecipe`. The choice of terminal vs SDK driver is orthogonal to what the
recipe is asking the judge to do. Drivers are passed to `judge_episode(...,
driver=...)` and overridden via the CLI's `--driver` flag. This also kills
the need for a `DRIVER_REGISTRY` to round-trip the recipe through JSON: the
recipe never references a driver, so JSON serialisation is straightforward
Pydantic.

Initial catalog (in `use_cases/`):

| Use case | When | Notable settings |
|---|---|---|
| `general_blame` | Default. Mirrors today's behaviour from #366. | Stock `JUDGE_SYSTEM_PROMPT`, full taxonomy, claude-sonnet-4-6. |
| `profiling` | Activated when an experiment is flagged with `profiling=True`. The judge focuses on a profiling trace path listed in the context file. | Adds `BashOutput` to allowed tools (so the judge can run helpers in `scripts/summarise_profile.py`); narrower taxonomy: `agent_scaffolding`, `model_capability`, `none`. |
| `agent_scaffolding` | Deeper blame ontology aimed at agents themselves (loop subtype, stuck phase, response-vs-action mismatch). | Output schema extends `JudgeOutput` via a sibling `scaffold_diagnosis` field — non-breaking, omitted by other use cases. |

`hint_harvest` and `auto_verified` are deferred to follow-up PRs (one
directory per PR) as agreed in design review.

### Context file format and requirement

The meta-agent generates one markdown file per experiment, by default at
`<experiment_dir>/judge_context.md`. The file MAY contain prose for the
judge's human readers but the only machine-parsed content is fenced code
blocks named `paths`:

````markdown
# Judge context for experiment 20260511_workarena_genny_h45

This experiment compares Genny (haiku-4-5) against the new Genny+memory
variant on WorkArena-L1. Focus on `agent_scaffolding` blames in the explore
phase.

```paths
cube_package: /Users/.../cube-harness/cubes/workarena
agent_package: /Users/.../cube-harness/src/cube_harness/agents
cube_harness: /Users/.../cube-harness/src/cube_harness
profiling_trace: /Users/.../runs/20260511_.../profiling.json
```
````

`validate_context_file(path)` already exists in `analyze/judge/context.py`
(seamed). It parses the `paths` fence, resolves relative paths against the
file's directory, raises `FileNotFoundError` if any path does not resolve,
and returns a `name -> Path` mapping.

**Critical change from revision 1:** the judge **requires** a context file.
It does not fall back to `collect_source_paths`. If
`<experiment_dir>/judge_context.md` is missing or any referenced path does
not resolve, the judge raises. There is no silent reconstruction path. The
rationale is that an ad-hoc judgment with mis-resolved venv paths is a class
of bug we have already paid for; making the context an explicit, validated
input keeps the judge's behaviour predictable across machines.

To preserve the developer ergonomics that the previous fallback supplied, a
new CLI subcommand bootstraps the file:

```
ch-judge init-context <experiment_dir>
```

This walks the experiment's `experiment_config.json` exactly as the old
`collect_source_paths` did, writes a `judge_context.md` with the discovered
paths, and exits. Developers run it once after an experiment finishes;
re-running the judge then uses the same file the meta-agent would have
produced. The file is committed-when-meaningful and inspected in PR review.

### Coding-agent driver Protocol

Today's `_run_claude_code` is one of N possible ways to drive a coding agent.
Auto-CUBE introduces an `AgentDriver` Protocol so the judge becomes a thin
"build prompt, parse output" layer over whichever driver is supplied.

```python
class DriverResult(TypedBaseModel):
    output_text: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float            # 0.0 when the driver has no metered billing
    duration_s: float
    actions: list[ToolAction]  # tool calls observed during the run

class ToolAction(TypedBaseModel):
    tool: str
    input_summary: str
    raw_input: dict[str, Any] | None   # populated only in trace_mode="full"

class AgentDriver(Protocol):
    name: str                  # e.g. "claude-code-sdk", "claude-terminal"
    max_parallelism: int       # advisory ceiling for `judge_experiment(n_parallel=...)`

    async def run(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        cwd: Path,
        additional_dirs: list[Path],
        model: str,
        allowed_tools: Sequence[str],
        verbose: bool = False,
        trace_mode: TraceMode = "actions",
    ) -> DriverResult: ...

    async def continue_session(
        self,
        *,
        follow_up_prompt: str,
        verbose: bool = False,
        trace_mode: TraceMode = "actions",
    ) -> DriverResult: ...
```

The first method signature mirrors `_run_claude_code` so the SDK refactor is
mechanical. `continue_session` is the seam used by the audit pass — it
appends a follow-up turn to the same conversation as the most recent `run`
call, so the audit prompt sees the prior judgment's context for free. A
driver that cannot continue a session (e.g. a stateless adapter) raises
`NotImplementedError` from `continue_session`; the judge catches that and
falls back to a fresh `run` with the prior judgment serialised into the
prompt.

`max_parallelism` is an advisory ceiling — `judge_experiment` clamps
`n_parallel` against `driver.max_parallelism` and emits a single log line on
clamp. It is advisory rather than enforced because cluster-level concurrency
caps live above this layer.

#### `ClaudeCodeSDKDriver`

Wraps today's `claude-agent-sdk` calls. `max_parallelism = 8` matches the
SDK's empirically stable parallel ceiling on a single host; higher numbers
hit `claude` CLI process churn. Cost is reported through `ResultMessage.usage`
exactly as today.

#### `TerminalClaudeDriver`

Launches `claude` via `asyncio.subprocess.create_subprocess_exec` with the
system prompt on the command line, the user prompt on stdin, and the
working directory set to `cwd`. Captures stdout/stderr, scrapes a final
JSON block via the same `_extract_json_block` helper the SDK path uses.

Trade-offs that drove the design:

- **Token accounting is unavailable.** `cost_usd` and per-call token counts
  are set to `0`, and `JudgeMetadata.cost_usd` becomes a lower bound rather
  than a true sum across an experiment. Documented; the joint CSV gains a
  `driver` column so downstream reports can scope cost summations to
  metered drivers.
- **Parallelism is low.** `max_parallelism = 2`. The terminal CLI uses a
  shared on-disk session store and is not safe to launch in higher fan-out;
  two is what the user's machine handles reliably. The clamp warning gives
  researchers a clear reason for any slowdown.
- **Tool surface is the CLI's, not the SDK's.** The driver passes
  `--allowed-tools` through but cannot disable tool categories the CLI
  doesn't expose. Same allowlist (`Read`, `Glob`, `Grep`, `Bash`) works for
  V1; recipes that need anything else fall back to the SDK driver.
- **No streaming progress.** `verbose=True` for the terminal driver streams
  the CLI's stderr unchanged. The "one line per tool call" formatting that
  the SDK driver produces is unavailable.
- **`continue_session` uses `claude --continue`** with the follow-up prompt
  on stdin. Works for the audit pass on a single host; not safe under
  parallel use of the same session store, so the audit pass on the terminal
  driver runs serially per host.

The trade is intentional: a subscription holder loses cost reporting and
parallelism but gains the ability to run judges at all without an API key.
Both drivers expose `.raw` on the returned object the caller can use for
escape-hatch inspection (`SR-003`), but the Protocol does not require it
because the raw types differ.

#### Codex driver (out of scope, signposted)

Codex would slot in as a third implementation of `AgentDriver`. Its
session-cookie auth and CLI flags are different enough to want its own
proposal. The Protocol's `additional_dirs` and `allowed_tools` already
accommodate the surface; nothing in V1 needs to change to admit it later.

### Constitution alignment: PS-002 (LiteLLM as the standard)

The constitution's PS-002 rule says LLM calls must go through LiteLLM, not
direct provider SDKs. The judge's `claude-agent-sdk` call is not a classic
LLM call — it is an *agent runtime* — but the question of "is there a standard
abstraction we should be using" still applies.

**There is.** LiteLLM ships first-party support for the Claude Agent SDK
([docs](https://docs.litellm.ai/docs/tutorials/claude_agent_sdk)) via its
proxy: the SDK is pointed at `ANTHROPIC_BASE_URL=<litellm-proxy>` with a
LiteLLM-issued auth token, and from then on every call the SDK makes is
routable through LiteLLM's provider catalog (Anthropic direct, Bedrock,
Vertex, Azure, …). LiteLLM also has a broader [Agent SDKs
page](https://docs.litellm.ai/docs/agent_sdks) covering related runtimes.

Auto-CUBE's drivers honour PS-002 by **routing through LiteLLM when a proxy
is configured**:

- `ClaudeCodeSDKDriver` reads `LITELLM_PROXY_URL` from the environment. When
  set, it exports `ANTHROPIC_BASE_URL=$LITELLM_PROXY_URL` and the matching
  auth header to the SDK before invoking it. When unset, it falls back to
  the SDK's default (direct Anthropic). This matches the integration pattern
  in the LiteLLM docs and keeps API-key-based developer flows unchanged.
- `TerminalClaudeDriver` exports the same env vars to the subprocess. The
  CLI honours them identically.

This is additive: no existing call site changes. The constitution rule is
satisfied because every billable token now flows through the canonical
gateway, and the driver Protocol stays small (no LiteLLM types leak into
`AgentDriver`).

A minor addition to the experiment record: when a LiteLLM proxy is in use,
the driver records `litellm_proxy_url` (URL only, no credentials) on
`DriverResult`, surfaced in `judge_metadata.json`. This is an audit signal
for "did this judgment route through the gateway?" and supports the
hermetic-reproducibility goal of PS-001.

If at follow-up time we discover LiteLLM's Agent SDK support drifts (model
list lag, missing parameters), the driver Protocol gives us the seam to fall
back to a thin internal wrapper without touching call sites. PS-002's
preferred order — "use LiteLLM if it covers the SDK; otherwise use another
mature standard; otherwise wrap" — is preserved by the Protocol regardless
of which path we sit on.

### Related-trajectory selection

A `Selector` is a small typed callable that returns the related-trajectory
paths the judge will be given alongside the main trajectory:

```python
class Selector(Protocol):
    name: str
    k: int                       # upper bound on returned trajectories

    def select(
        self,
        *,
        main_episode: EpisodeRef,
        experiment_dir: Path,
        all_refs: Sequence[EpisodeRef],
    ) -> list[Path]: ...
```

Built-in selectors:

- `SameTaskDifferentAgent(k=3)` — returns up to `k` episodes from sibling
  experiments that share the same `task_id` but differ on
  `agent_config._type`. Useful for contrastive analysis ("Genny took 30
  steps, ReAct took 12, why?").
- `SameAgentPreviousIteration(k=2)` — within the same experiment family
  (resolved via the experiment-level `family_id` already written into
  `experiment_config.json`), returns up to `k` predecessors with the same
  task. Useful for hypothesis-tracking iterations.
- `TopKBySimilarityStub(k=3)` — placeholder that returns the `k` most
  recent neighbours by directory mtime. The intent is to swap in a real
  similarity scorer (trajectory length + outcome + tool-call profile)
  without changing the selector contract.

Selectors live in `analyze/judge/selection.py` alongside the existing
`EpisodeRef` / `discover_episodes` / `select_episodes` helpers, which are
about *which episodes to judge in the current experiment*. Naming is
deliberate: episode *selection* picks the work pool; related-trajectory
*selection* picks neighbours per piece of work. Both are operations on
`EpisodeRef`; sharing the module keeps the file boundary aligned with the
domain.

A selector is supplied per call (`judge_episode(..., selector=...)`) rather
than baked into the recipe, mirroring the driver decision: which neighbours
to pull is an outer-loop policy choice the meta-agent makes, not a property
of the judging task itself.

### Post-judge survey

After the judge writes its `JudgeOutput`, a second pass runs against the
same driver, with a short system prompt that asks the judge to fill in
`post_judge_survey.json`. The schema is fixed:

```python
class PostJudgeSurvey(TypedBaseModel):
    schema_version: int = 1     # mirrors JUDGE_SCHEMA_VERSION on JudgeMetadata
    recipe: str
    ease_of_analysis: int = Field(..., ge=0, le=5)
    context_quality: int = Field(..., ge=0, le=5)
    tooling_gaps: list[str] = Field(default_factory=list)
    notes: str | None = None
```

`schema_version` is on the model (not in the filename) to mirror the existing
`JudgeMetadata.JUDGE_SCHEMA_VERSION` convention, so survey schema and judge
schema evolve through the same mechanism. Filename versioning would diverge
from the existing pattern with no win.

The seam PR lands the stub: today, the file is written with `None` integers
and an empty list. Auto-CUBE populates it. Three design points:

- **Why a second pass and not a single richer schema?** Survey questions
  pull the judge toward meta-reasoning ("how easy was this?") that
  contaminates the primary judgment. Splitting the passes keeps `JudgeOutput`
  free of self-referential noise.
- **Why not an embedding-based heuristic instead of asking the model?**
  Tooling-gap detection ("I wanted to see the screenshot but the file was
  missing") is inherently semantic. A heuristic that grep'd for "I couldn't
  find" would catch maybe 40% of cases.
- **Cost.** Roughly +15% on the judgment's total cost. The
  `JudgeRecipe.post_judge_survey` flag lets recipes that don't care skip it.

The survey is read by the audit pass and by the eventual sweep aggregator
that produces `tooling_gaps.csv`.

### Audit pass

Replaces the previous `self_judge` recipe. The shape:

- **Trigger.** `audit=True` on the recipe, or `--audit` at the CLI / via
  `judge_experiment(audit=True)`. Default off.
- **Mechanism.** After the primary judgment writes `judge_output.json` and
  the survey writes `post_judge_survey.json`, the judge issues
  `driver.continue_session(follow_up_prompt=AUDIT_PROMPT)`. The audit prompt
  asks the judge to critique its own reasoning: were the cited evidence
  pieces actually load-bearing? Did the judge skip a contradicting tool
  call? Would a different `primary_blame` be defensible?
- **Output.** `audit.json` next to `judge_output.json`. Schema:

  ```python
  class AuditOutput(TypedBaseModel):
      schema_version: int = 1
      recipe: str
      driver: str
      reasoning_quality: int = Field(..., ge=0, le=5)
      missed_evidence: list[str] = Field(default_factory=list)
      alternative_blames: list[BlameAlternative] = Field(default_factory=list)
      verdict: Literal["sound", "questionable", "wrong"]
      notes: str | None = None

  class BlameAlternative(TypedBaseModel):
      blame: str
      rationale: str
  ```

- **Driver fallback.** When the driver does not implement
  `continue_session`, the audit pass calls `driver.run(...)` afresh with the
  prior judgment, prior trace, and prior survey serialised into the user
  prompt. Same `audit.json` output; the only loss is that the audit no
  longer benefits from in-conversation context economy.

The audit is **not** a recipe in the catalog because it always layers on top
of an existing judgment. Treating it as a flag avoids the temptation to
fork the catalog (`general_blame`, `general_blame_audit`,
`profiling_audit`, …).

The follow-up PR — out of scope here — adds the **batch audit step** the
user asked for: a meta-agent skill (`audit_batch.md`) plus a script
(`analyze/cross_experiment/audit_batch.py`) that walk a sweep directory,
collect every `audit.json`, re-judge any trajectory whose verdict is
`questionable` or `wrong`, and produce a per-sweep
`audit_quality_report.md`. This RFC fixes the on-disk schema for `audit.json`
so the batch tool can be written without churn.

### Cross-judge agreement

For a sweep where the meta-agent wants a confidence signal, the inner loop
is run multiple times per episode **with the same recipe at different
seeds**. Cross-recipe agreement is intentionally out of scope for this
artifact — comparing a `general_blame` judgment to an `agent_scaffolding`
judgment is well-defined only on shared core fields, and the user feedback
is that the more useful signal is single-recipe variance across seeds (does
the same recipe reach the same blame on the same trajectory twice?).

The runner that does the re-judging is out of scope for the first PR; the
**output shape** is fixed here so the joint-CSV consumer never breaks when
the runner ships.

For each (recipe, trajectory) pair that has more than one judgment, a row
is written to `<experiment_dir>/cross_judge_agreement.csv`:

| Column | Meaning |
|---|---|
| `trajectory_id` | the episode |
| `recipe` | the recipe name (single value — agreement is per-recipe) |
| `n_judgments` | how many seeds were run for this (recipe, trajectory) pair |
| `seeds` | semicolon-separated seed values |
| `primary_blame_modal` | the modal `primary_blame` across judgments |
| `primary_blame_agreement` | fraction in `[0, 1]` agreeing with the mode |
| `outcome_modal` | the modal `outcome` |
| `outcome_agreement` | fraction agreeing with the mode |
| `confidence_mean` | mean of `primary_blame_confidence` across judgments |

This file is the meta-agent's confidence signal. Low
`primary_blame_agreement` flags an episode for human review. It is also the
input to recipe-quality work: if a single recipe disagrees with itself
across seeds on a specific failure mode, that recipe's prompt needs
sharpening.

The choice to put cross-judge state in a sibling CSV rather than nesting it
inside `EpisodeRecord` mirrors the
`experiment_judge_summary.json` / `episode_record.judge_output` split from
PR #366: the per-episode record stays a pure analytical document; aggregate
artifacts live at the experiment root.

### Joint CSV across experiments

The outer loop concatenates per-experiment `experiment_judge_report.csv`
files into one row per (experiment, episode), with these added columns:

- `experiment_id` — the experiment directory name.
- `family_id` — the sweep this experiment belongs to (already written by
  the experiment runner; copied verbatim here).
- `agent_dotted` / `benchmark_dotted` — the dotted `_type`s pulled from
  `experiment_config.json`.
- `driver` — the driver name (`claude-code-sdk`, `claude-terminal`, ...).
- `recipe` — the recipe name (`general_blame`, ...).
- `litellm_proxy_url` — when set; empty string otherwise. Lets reports
  filter "judgments routed through the gateway" from "judgments that hit
  Anthropic direct."
- All columns from `experiment_judge_report.csv` (the per-experiment file
  produced by PR #366).
- Cross-judge agreement columns when available, joined on
  `(trajectory_id, recipe)`.

Written by `cube_harness.analyze.cross_experiment.joint_csv.write_joint_csv`
to `<sweep_dir>/joint_judge_report.csv`. CSV rather than Parquet because the
meta-agent reads it with `grep`/`csv.reader`; the row count rarely exceeds
the low thousands for a sweep. Parquet is a follow-up if and when sweep
sizes grow.

### Meta-agent integration

The meta-agent owns the outer loop. Its responsibilities, in the order they
fire:

1. **Plan the sweep.** Pick the experiments to dispatch. This stage is
   human-in-the-loop today and stays that way for V1 — a researcher writes a
   recipe under `recipes/` that defines the sweep, or invokes
   `/meta-agent` interactively.
2. **Generate context files.** For each experiment, write
   `<experiment_dir>/judge_context.md` with the paths and prose the judge
   needs. The meta-agent does this *before* the experiment runs and the
   context file becomes part of the experiment's artifacts. Required, not
   optional.
3. **Dispatch.** Submit experiments via the existing
   `run_sequentially` / `run_with_ray` path. The meta-agent does not touch
   the episode loop.
4. **Judge.** When each experiment finishes, the meta-agent kicks
   `judge_experiment` with the recipe selected for that experiment and the
   driver chosen for the cluster (typically `claude-code-sdk` on shared
   infra; `claude-terminal` on a researcher's laptop). Recipe selection is
   rule-based for V1, driven by the use-case directories' `SKILL.md`
   metadata.
5. **Aggregate.** Write `joint_judge_report.csv` for the sweep. Optionally
   re-judge for cross-judge agreement (same recipe, multiple seeds).
6. **Analyse.** Produce an `experiment_judge_report.md` per experiment and a
   `sweep_judge_report.md` for the whole sweep. These are LLM-authored
   markdown documents with cited evidence from the joint CSV. The meta-agent
   is the author; this RFC does not specify the prompt.
7. **Audit (optional, follow-up).** Run the audit-batch step over the
   previous sweep — collect all `audit.json` files, re-judge flagged
   trajectories, produce `audit_quality_report.md`. Schema fixed in this
   RFC; runner ships in the follow-up.
8. **Intervene.** Propose interventions: prompt edits, scaffold tweaks, new
   hints, recipe changes. **All interventions are reviewed by a human
   before merge.** This is the human-in-the-loop boundary: the meta-agent
   can author, dispatch, judge, and report; it cannot push to a branch or
   merge a PR.

The boundary matters. A self-judging system that ships its own fixes is a
much higher-risk surface than one that ships proposals. Auto-CUBE is the
latter.

### Migration of `meta_agent/`

The `meta_agent/` directory today holds `recipes/` (one file:
`workarena_l1_full.py`), `hints/` (one file:
`swebench-verified.json`), the standalone script `workarena_hints.py`, and a
`README.md`. The skill at `.claude/commands/meta-agent.md` and the per-session
journals under `~/cube_meta_agent_journal/` are out of scope for this RFC.

The migration is conservative because there is not much to move:

| Today | Auto-CUBE | Reason |
|---|---|---|
| `meta_agent/recipes/workarena_l1_full.py` | stays in place | These are *experiment* recipes (Pydantic `Experiment` instances), not *judge* recipes. They are the outer-loop dispatch artifacts. |
| `meta_agent/hints/swebench-verified.json` | `cubes/swebench-verified/hints/` (per-cube) | Hints are consumed by the benchmark code at runtime, not by the meta-agent. Per-cube ownership matches the existing pattern documented in `meta_agent/README.md`. |
| `meta_agent/workarena_hints.py` | `cubes/workarena/scripts/` or deleted | One-off script; not part of the loop contract. |
| Future judge use cases | `cube_harness/analyze/judge/use_cases/<name>/` | New home. Each use case is a directory with recipe + skill + scripts. |
| Future cross-experiment tools | `cube_harness/analyze/cross_experiment/` | New module for the joint-CSV, cross-judge, and audit-batch aggregators. |
| Slash command at `.claude/commands/meta-agent.md` | stays | Until the dispatch / analysis loop is codified into typed Python, the slash command is the runner. Its `SKILL.md`s are now sourced from `use_cases/<name>/SKILL.md`. |

The journals under `~/cube_meta_agent_journal/` remain machine-local; this
RFC does not propose committing them.

### Trace and telemetry surface

Every Auto-CUBE call sits on the existing OTel tracer (`SR-002`). The driver
Protocol's `run` and `continue_session` are wrapped in a
`tracer.span("auto_cube.judge.driver_run")` (or `..._continue`) with
attributes:
- `auto_cube.driver` — driver name
- `auto_cube.recipe` — recipe name
- `auto_cube.model` — model string
- `auto_cube.litellm_proxy` — boolean, true when `LITELLM_PROXY_URL` was set
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens` from the driver
  result (zero on the terminal driver)

This is purely additive — no spans are removed, no attribute names change.

---

## Trade-offs

### TerminalClaudeDriver parallelism

The driver caps itself at `max_parallelism = 2`. This is a real cost: an
experiment with 50 episodes that the SDK driver judges in 7 minutes at
`n_parallel=8` takes ~25 minutes through the terminal driver. The
alternative — running terminal sessions at higher fan-out — corrupts the
shared on-disk session store and produces empty judgments. We pay the
latency for users who do not have an API key. The recipe catalog flags this
in a docstring on `TerminalClaudeDriver` and the joint CSV's `driver` column
gives downstream reports a way to scope cost analyses to metered drivers.

### Use-case directory churn

Three use cases at launch (`general_blame`, `profiling`, `agent_scaffolding`)
is small enough to keep the directory layout legible. The risk is that every
new analytical question proposes a new use case directory and the catalog
drifts to dozens of near-duplicates. Three guardrails:

- Use cases are typed Python directories. A new use case must add a
  directory with `recipe.py`, `SKILL.md`, and (optionally) `prompts/` and
  `scripts/`. The cost of adding one is visible in a PR diff.
- The taxonomy in `JudgeOutput` is shared by all use cases. They can extend
  with sibling fields (as `agent_scaffolding` does with `scaffold_diagnosis`)
  but cannot fork the core taxonomy. Aggregation in the joint CSV stays
  meaningful.
- The post-judge survey's `tooling_gaps` field is the signal for whether a
  new use case is needed vs. an existing one needs sharpening. If a gap
  repeats across episodes for an existing use case, sharpen it; if it
  repeats across use cases, propose a new one.

### Judge cost vs. coverage

Cross-judge agreement multiplies judge cost by the number of seeds per
episode. A 50-episode experiment judged at `n_seeds=3` costs roughly 3× a
single-judgment baseline. The first PR does not ship the cross-judge
runner; when it does, the meta-agent will run cross-judge only on episodes
flagged by low primary-blame confidence (≤ 2) from the first pass. This
keeps the cost bounded to the episodes where the signal is worth the spend.

### Survey overhead

The survey adds roughly 15% to judgment cost per episode. The trade is the
feedback loop into tooling improvements. Use cases that don't need it skip
the pass via `post_judge_survey=False`; the default in V1 is on.

### Audit overhead

The audit pass adds roughly 25% to per-episode cost (one extra LLM turn that
re-reads the prior judgment). Off by default. When on, the meta-agent gets
a per-episode quality signal that the survey alone does not provide
(survey is the judge's self-report on how easy the analysis was; audit is
the judge's critique of its own conclusion).

---

## Phasing

### First Auto-CUBE PR (this RFC's deliverable)

- `AgentDriver` Protocol in `cube_harness/analyze/judge/driver.py`, including
  `continue_session` for the audit pass.
- `ClaudeCodeSDKDriver` — refactor of today's `_run_claude_code` into the
  Protocol shape; bit-identical behaviour for the `general_blame` recipe.
  Honours `LITELLM_PROXY_URL` per PS-002.
- `TerminalClaudeDriver` — subprocess implementation, capped at
  `max_parallelism=2`. Honours `LITELLM_PROXY_URL`.
- `JudgeRecipe` widening: `user_prompt_template`, `audit`,
  `post_judge_survey`, `permission_mode` fields. **No `driver` field** —
  drivers are call-time, not recipe-time.
- Use-case directories under `cube_harness/analyze/judge/use_cases/`:
  `general_blame`, `profiling`, `agent_scaffolding`. Each contains
  `recipe.py`, `SKILL.md`, `prompts/`, and (for `profiling`) `scripts/`.
- `RECIPE_CATALOG` assembled by `use_cases/__init__.py` walking
  subdirectories.
- `Selector` Protocol + the three built-in selectors. Selector is supplied
  per call, not on the recipe.
- Survey collection: second-pass driver invocation; schema enforcement with
  `schema_version`.
- Audit pass: flag-gated; `audit.json` schema with `schema_version`.
- Context-file as the **only** input to the judge (`validate_context_file`
  promoted from "fallback when used" to "required, raises if missing"). New
  CLI subcommand `ch-judge init-context` for ad-hoc bootstrap.
- `joint_judge_report.csv` schema fixed (no runner yet).
- `cross_judge_agreement.csv` schema fixed (no runner yet).
- `meta_agent/` migration: directory layout updated, README rewritten,
  `hints/swebench-verified.json` moved to `cubes/swebench-verified/`.

### Follow-up PRs (called out in this RFC)

1. **Cross-judge runner** — the `judge_experiment` extension that runs
   episodes through the same recipe at multiple seeds and writes
   `cross_judge_agreement.csv`. Gated on `primary_blame_confidence ≤ 2` to
   keep cost bounded.
2. **Joint-CSV runner** — the `write_joint_csv(sweep_dir)` helper plus the
   sweep-discovery walk.
3. **Audit-batch runner** — the meta-agent skill +
   `analyze/cross_experiment/audit_batch.py` script that walks a sweep,
   collects `audit.json` files, re-judges flagged trajectories, and writes
   `audit_quality_report.md`.
4. **Codex driver** — third `AgentDriver` implementation, also
   LiteLLM-routable via the Codex Agent SDK adapter.
5. **TopK similarity selector** — replace the mtime stub with a real
   trajectory-feature scorer.
6. **Additional use cases** — one PR per directory: `hint_harvest`,
   `auto_verified`, and any new ones the survey/audit signal motivates.
7. **Typed meta-agent loop** — promotes the slash command's responsibilities
   into `cube_harness.analyze.cross_experiment.meta_agent` with typed
   dispatch and analysis steps. The slash command stays as the human-facing
   entry point.

---

## Open questions

1. **Where does the per-use-case `SKILL.md` get registered?** The Claude
   Code skill convention expects skills under `.claude/commands/` (or
   `.claude/skills/`). The use-case directory layout puts the source file
   under `analyze/judge/use_cases/<name>/SKILL.md`. V1 plan: a small script
   committed alongside this RFC creates symlinks at
   `.claude/skills/judge-<name> -> analyze/judge/use_cases/<name>/SKILL.md`.
   Is the symlink approach acceptable, or do we want a registration step
   that copies them?
2. **`continue_session` semantics on the SDK driver.** The
   `claude-agent-sdk` exposes a continuation API; we want the audit pass to
   reuse the same session so prior context is in-conversation. Open: does
   the SDK's continuation handle the case where the session has crossed
   tool-call boundaries (the survey pass between the primary judgment and
   the audit pass), or do we need to checkpoint the session ID through
   `DriverResult` and restore it manually? Verifying against the SDK before
   the first PR ships.
3. **Audit prompt design.** This RFC fixes the schema and the trigger but
   leaves the audit prompt unspecified. The prompt belongs in the
   `general_blame` use-case directory's `prompts/audit.md` (so each use
   case can override). V1 ships a single shared `prompts/audit.md` symlinked
   into each use case until we see divergence.
4. **`init-context` defaults.** `ch-judge init-context <experiment_dir>`
   walks `experiment_config.json` and writes a starter file. Open: does it
   include a prose section template, or just the `paths` fence? V1 plan:
   paths fence only — the meta-agent fills prose when it generates the file
   for real; the developer-bootstrap form has empty prose.
5. **Cross-judge cost gating.** The follow-up cross-judge runner targets
   episodes with `primary_blame_confidence ≤ 2`. Is the threshold the right
   one (some recipes systematically run high-confidence even when wrong),
   or do we need a use-case-specific threshold? Defer to the audit-batch
   PR, which will have data to calibrate against.

---

## References

- `openspec/changes/trajectory-judge/proposal.md` — PR #366, the inner-loop
  judge this RFC extends.
- `openspec/changes/trajectory-judge/deltas.md` — the V1 judge surface area.
- `openspec/changes/atlas-eval-log/proposal.md` — `EpisodeRecord` and the
  `judge_output` / `judge_metadata` fields the inner loop writes.
- `openspec/changes/agent-owns-loop/proposal.md` — orthogonal episode-loop
  work; does not touch the judge.
- `.claude/commands/meta-agent.md` — the current slash command driving the
  outer loop.
- `.claude/commands/judge-traces.md` — the slash command for invoking the
  judge directly.
- `src/cube_harness/analyze/judge/` — the judge package on the
  `feat/trajectory-judge` branch (PR #366), where the seam PR landed
  `JudgeRecipe`, `PostJudgeSurvey`, `validate_context_file`, and the
  `related_trajectories` parameter that this RFC builds on.
- `meta_agent/` — the existing outer-loop scaffolding.
- `.claude/rules/constitution.md` — the principles this RFC honours
  (Python-as-config; composition over inheritance; LiteLLM via the
  Anthropic env-var path on the SDK; trace-first; escape hatches on
  drivers).
- LiteLLM Claude Agent SDK integration:
  https://docs.litellm.ai/docs/tutorials/claude_agent_sdk
- LiteLLM Agent SDKs overview:
  https://docs.litellm.ai/docs/agent_sdks
