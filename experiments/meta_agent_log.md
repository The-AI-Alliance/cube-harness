# Meta-Agent Log — swebench-verified-cube

Branch: `worktree-meta-agent-swe-debug` (stacked on `feat/meta-agent-3-structural` + merged #315 episode-status).

## Iteration 1 — 2026-04-28 — Make `hello_swebench_verified.py` actually run on Daytona

**Tasks**: any (recipe-level fix; no per-task signal needed)

**What was broken**:
- Recipe imported `cube.backends.daytona.DaytonaContainerBackend` — that module's
  internal import of `cube_infra_daytona` fails because `cube-infra-daytona` is
  not declared as a dependency anywhere reachable from the recipe.
- Recipe constructed `SWEBenchVerifiedBenchmark(container_backend=backend)` —
  the benchmark class only declares `infra: InfraConfig`, so the recipe's
  intent (run on Daytona) was silently dropped and the cube fell back to
  `LocalInfraConfig`.
- Recipe didn't load `~/.env-cube` / `~/.env`, so `OPENAI_API_KEY` (or any LLM
  credential outside the shell environment) was missing.

**Hypothesis**: The recipe was authored against an older API and hasn't been
revisited since the `BenchmarkConfig` / `InfraConfig` split landed. There is no
working non-local recipe for this cube today.

**Intervention**: Tried to import `DaytonaContainerBackend` in a one-liner — got
`ModuleNotFoundError`. Read benchmark.py — confirmed `infra: InfraConfig` is the
only field. `DaytonaInfraConfig` exists in `cube-standard/cube-resources/cube-infra-daytona/`
but isn't on PyPI and isn't a declared dep.

**Fix** (`recipes/hello_swebench_verified.py`):
- Add `cube-infra-daytona` (editable, sibling cube-standard checkout) and
  `python-dotenv` to inline-script deps.
- Replace `from cube.backends.daytona import DaytonaContainerBackend` with
  `from cube_infra_daytona import DaytonaInfraConfig`.
- Replace `SWEBenchVerifiedBenchmark(container_backend=backend)` with
  `SWEBenchVerifiedBenchmark(infra=DaytonaInfraConfig())`.
- `load_dotenv(~/.env-cube)` per #314, then `load_dotenv(~/.env, override=False)`
  as fallback — most users (including the project owner today) still keep
  credentials in `~/.env`.

**Result**: Recipe imports cleanly, `infra=daytona:us` reported in the logs,
sandbox launches per task, first LLM call (azure/gpt-5-mini) succeeds, episodes
advance.

**Blast radius**: One file, recipe-only. No library code changed. Deferred-to-PR
notes:
- The hardcoded absolute path `/Users/alexandre.lacoste/...` in `[tool.uv.sources]`
  is for this worktree only (5-level relative path differs from main checkout).
  Restore to `../../cube-standard/cube-resources/cube-infra-daytona` before any PR.
- The pattern of editable deps on a sibling cube-standard checkout is fragile.
  Long-term fix is publishing `cube-infra-daytona` to PyPI or making
  `cube-standard[daytona]` actually pull it in. Out of scope for this branch.

## Iteration 2 — 2026-04-28 — Tool param descriptions (units, semantics)

**Tasks**: `django__django-10097` (run #2)

**What the agent saw**: At turn 3 the agent emitted
`bash(command="grep -Rn ...", timeout=120000)`. 120000s = 33 hours. The schema
the agent received had **no description** on the `timeout` parameter, just
`{'timeout': {'type': 'integer'}}`. The LLM defaulted to a millisecond
convention common in many APIs.

**Hypothesis**: Tool docstrings in `cubes/swebench-verified-cube/tool.py` had
only a top-line summary, no `Args:` block. `cube.utils.function_to_dict` parses
Google-style docstrings to fill the JSON-schema parameter `description` field;
without `Args:` blocks the parameters appear bare. There is a
`ActionSchema.validate_param_descriptions` validator in cube-standard but it is
never auto-called.

**Intervention**: None needed — the absent descriptions are directly visible in
the schema dump from `llm_call.prompt.tools`.

**Fix** (`cubes/swebench-verified-cube/src/swebench_verified_cube/tool.py`):
add `Args:` blocks to `bash`, `read_file`, `write_file`. Specifically:
- `bash.timeout`: explicit "wall-clock seconds, default 120, NOT milliseconds"
- `bash.command`: working dir is /testbed
- `read_file.path` / `write_file.path`: relative paths resolve against /testbed
- `write_file.content`: clarify it's a full overwrite, not a patch

**Result** (pending — applied mid-run, takes effect on next invocation): TBD.
Validation will be: re-run, confirm `tools` payload now contains
`{'timeout': {'description': 'Wall-clock seconds...', 'type': 'integer'}}`,
and watch for unrealistic timeouts.

**Follow-up candidate**: `ActionSchema.validate_param_descriptions` should be
called automatically on tool registration / `action_set` access in
cube-standard, so missing descriptions fail loudly instead of silently shipping
to the LLM. That belongs in cube-standard, not here.
