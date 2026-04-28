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

## Iteration 3 — 2026-04-28 — Recipe never reached `max_steps=30`; agent capped at 10

**Tasks**: `django__django-10097` (run #2)

**What we observed**: Episode 0 ended at turn 11 with reward 0.0 and the log line:

> `cube_harness.agents.react:74 step() - Max actions reached, issuing STOP action.`

The recipe sets `Experiment.max_steps=30` but never set `ReactAgentConfig.max_actions`,
which defaults to **10** in `src/cube_harness/agents/react.py:19`. The smaller of
the two limits wins; the agent was force-stopped at action 10 and the harness
fired `final_step` for it.

This means the recipe has been silently giving every SWE-bench task a 10-action
budget — far below what real-world bug fixes need (explore → read → patch →
verify ≈ 20–40 actions). Every `reward=0.0` we'd ever seen on this cube was
under-resourced.

**Hypothesis**: Two competing limits exist. The `Experiment.max_steps` is the
episode-wall limit; `AgentConfig.max_actions` is an agent-internal counter that
short-circuits earlier. Recipes need to set both, or the agent ones default low.

**Intervention**: Skipped — the log line is unambiguous and the default value
in `react.py` matches.

**Fix** (`recipes/hello_swebench_verified.py`): set
`max_actions=30` on the `ReactAgentConfig` to match `max_steps`, with a
comment so the next reader doesn't strip it.

**Result**: Confirmed in run 3 — episode reached turn 12 (up from 10). Agent called `final_step` at
turn 11 with reward 0.0. The budget fix works; agent stopped from logic error, not from the cap.

**Follow-up candidates** (out of scope for this branch, but worth noting):
- `ReactAgentConfig.max_actions` default of 10 is too low for any
  multi-step coding/reasoning task. Either raise the default to match common
  use, or remove the agent-side cap and let the experiment-level
  `max_steps` be the single source of truth.
- The harness should warn (or refuse to start) when
  `agent.max_actions < experiment.max_steps`, since the smaller value is
  almost always a misconfiguration.

## Iteration 4 — 2026-04-28 — evaluate() output truncated at 2000 chars hides test failures

**Tasks**: `django__django-10097` (run #3)

**What we observed**: The `fail_to_pass_output` in the episode metadata was exactly 2000 characters,
cutting off in the middle of Django's test runner startup output (DB creation logs). The actual
test failures (FAIL / ERROR lines) appeared after 2000 chars and were invisible.

**Hypothesis**: `task.py:evaluate()` truncates `f2p_output[:2000]` before storing. Django's
runtests.py emits lengthy DB-setup preamble before printing test results, so 2000 chars is
always too short to see any test outcome.

**Intervention**: None needed — the truncation is directly visible at the cut-off character.

**Fix** (`cubes/swebench-verified-cube/src/swebench_verified_cube/task.py`):
Increase truncation from 2000 → 20000 chars for both `fail_to_pass_output` and `pass_to_pass_output`.

**Result**: Confirmed diagnostic — the agent's regex for django__django-10097 allowed `:` in the
password part (`[^\s@/]*`) but the gold patch requires `[^\s:@/]*`. Without seeing the truncated
output we could not verify which specific test case failed. With 20000-char limit the next run
will show the exact FAIL/ERROR lines.

**Follow-up candidate**: store full eval output to a sidecar file and only store a summary in
the info dict — avoids the size vs. visibility tradeoff entirely.

## Iteration 5 — 2026-04-28 — `read_file()` crashes on `line_start`/`line_end` kwargs

**Tasks**: `django__django-10554` (run #3, ep1)

**What the agent saw**: At turn 17 the agent emitted
`read_file(path="django/db/models/sql/query.py", line_start=640, line_end=1040)`.
`SWEBenchTool.read_file()` only accepts `path`, so `execute_action()` caught a `TypeError` and
returned `StepError`. `task.step()` treats `StepError` as `done=True`, terminating the episode
with `had_step_errors: True` and `reward=0.0`.

**Hypothesis**: The LLM is trained on Claude's own `read_file` tool (which accepts
`line_start`/`line_end` for windowed reading). Our implementation only accepted `path`, so any
attempt to read a file range caused an immediate fatal error.

**Intervention**: None needed — the TypeError is unambiguous in the episode log.

**Fix** (`cubes/swebench-verified-cube/src/swebench_verified_cube/tool.py`):
Add `line_start: int | None` and `line_end: int | None` parameters to `read_file`. When provided,
use `sed -n '{start},{end}p'` instead of `cat`. Both parameters are optional (defaults to full file).

**Result**: pending re-run.

**Blast radius**: `read_file` schema changes (new optional params), which is backwards-compatible.
Agents that don't pass line ranges are unaffected.

**Minor fix also committed**: `import re` moved from inside `_normalize_django_directive()` to
module-level (EX-001 violation).

## Iteration 6 — 2026-04-28 — Agent calls wrong test runner; fix never verified

**Tasks**: `django__django-10097` (run #3, ep0)

**What the agent saw**: At turn 17 the agent ran `python3 -m unittest tests.validators.tests -v`.
This failed with `ImproperlyConfigured: Requested setting DATABASES`. Django tests require either
`DJANGO_SETTINGS_MODULE` to be set or the `runtests.py` launcher. The agent's inline validation
scripts passed (correct URLValidator behaviour) so it concluded the fix was right and called
`final_step` — but the eval used `./tests/runtests.py` and the actual test cases failed (agent's
regex allowed `:` in the password part; gold requires `[^\s:@/]*` in both halves).

**Hypothesis**: The system prompt said "when you are confident the fix is correct, call final_step"
but gave no guidance on *how* to run tests in this repo. The LLM defaulted to `python -m unittest`
which is the generic Python pattern — not the Django runtests.py approach used during evaluation.

**Intervention**: Confirmed by reading the agent's episode log turn-by-turn.

**Fix** (`recipes/hello_swebench_verified.py` — `SWE_SYSTEM_PROMPT`):
Add framework-specific test commands:
- Django → `./tests/runtests.py --verbosity 2 <module>`
- SymPy → `./bin/test`
- Other → `python -m pytest`
and note that `python -m unittest` does NOT work for Django.

**Result**: pending re-run (run 4 in progress).

**Blast radius**: Recipe-only (system prompt). No library code changed.

## Iteration 7 — 2026-04-28 — evaluate() output preamble drowns test results even at 20000 chars

**Tasks**: `django__django-10097` (run #4, ep0)

**What we observed**: With the 20000-char `[:20000]` limit from iter 4, the stored F2P output still
showed only "Cloning test database for alias 'other'..." — Django runs 128 parallel workers, each
cloning a DB, producing tens of thousands of preamble lines before any test results appear.
Switching to `[-20000:]` showed the same content — the preamble fills the entire output.

**Hypothesis**: The setup preamble (DB creation + parallel worker cloning) for 438 tests with 128
workers is ~100k+ chars. Storing any fixed prefix or suffix of the raw output is unreliable;
the test result lines are sandwiched between preamble and summary in the middle.

**Fix** (`_run_tests()` in `task.py`):
Pipe `{ test_cmd 2>&1; echo CUBE_TEST_EXIT_CODE:$?; } | tail -n 200` to keep only the last 200
lines. Embed the actual exit code in a sentinel line before the tail so the pipeline exit code
(always 0 from `tail`) doesn't mask failures. Regex-extracts and removes the sentinel line before
returning the output.

**Result**: pending run 5 (first run with the tail fix active).

**Blast radius**: `_run_tests()` only. `all_passed` logic unchanged in semantics; sentinel approach
handles pass/fail/timeout correctly (verified by unit test in session).

## Iteration 8 — 2026-04-28 — Agent still passes timeout=120000 for bash commands

**Tasks**: `django__django-10554` (run #4, ep1)

**What we observed**: At T9/T13, the agent called `bash(command="grep ...", timeout=120000)`.
Despite the docstring saying "NOT milliseconds", the LLM defaults to millisecond-scale values from
its training data (120000ms ≈ 120s). The grep finishes in milliseconds regardless, so no
functional harm — but a test-suite call with timeout=120000 would wait 33 hours.

**Hypothesis**: The LLM's pattern-matching overrides the docstring. "NOT milliseconds" is easy to
miss especially since the LLM's prior association is strong (120000 is a common millisecond timeout).

**Fix** (`tool.py:bash()`): Silently cap timeout to 120s when the value exceeds 7200s. This
converts the agent's intended 120000ms → 120s without any visible error.

**Follow-up**: The description says "Use larger values (600-1800)" — perhaps 120000ms → 120s is
not what the agent wants for tests. Monitor whether agents use large timeouts for test commands
and whether the cap causes premature timeouts.

## Iteration 9 — 2026-04-28 — Switch debug tasks from Django to requests + flask

**Tasks**: django__django-10097, django__django-10554 (outgoing); psf__requests-1142, pallets__flask-5014 (incoming)

**What we observed**:
- django__django-10097: `fail_to_pass` contains 438 tests total; only 17 are URLValidator-related.
  The other 421 are pre-existing auth test failures unrelated to the patch. Any correct fix gets
  reward=0.0 because the 421 tests will always fail in this eval environment. No diagnostic value.
- django__django-10554: 2 clean fail_to_pass tests, but the bug is a deep compiler internals fix
  in `compiler.py:get_order_by()`. Agent explored the right file (step 19-23) then regressed to
  patching `query.py` for 20 more turns and never applied the correct fix. Out of reach for debug tasks.

**Hypothesis**: Debug task selection matters as much as pipeline quality. Tasks with noisy
`fail_to_pass` sets produce false-negatives on correct fixes. Tasks requiring deep compiler
knowledge exceed what small-scale debugging can diagnose.

**Fix** (`recipes/hello_swebench_verified.py`):
- Replace `DEBUG_TASKS` (was first 2 django tasks) with:
  - `psf__requests-1142`: 1 f2p, 5 p2p — "GET always sends Content-Length" — don't set
    Content-Length for GET/HEAD. Simple, isolated, clean signal.
  - `pallets__flask-5014`: 1 f2p, 59 p2p — "Empty Blueprint name should raise ValueError".
    One-line constructor guard. Both use plain pytest (no Django DB setup preamble).
- Add `--tasks` CLI argument for ad-hoc task overrides.
- Default model bumped to `azure/gpt-5.4`.

**Result**: Run 5 — `pallets__flask-5014` solved in 6 turns, reward=1.0 (first successful reward).
`psf__requests-1142` — reward=0.0 (new failure mode discovered, see Iter 10).

**Blast radius**: Recipe-only. No library code changed.

## Iteration 10 — 2026-04-28 — Agent returns no-action response instead of calling final_step

**Tasks**: `psf__requests-1142` (run 5)

**What the agent saw**: At turn 11 the agent's pytest run showed "4 passed, 23 deselected, 1 warning in 0.04s"
— the fix was correct. But instead of calling `final_step`, the agent returned a plain text response with
no tool calls. `episode.py:213` detects this, logs "Agent returned no actions — stopping episode." and
`break`s out of the loop, using the last `env_output` (done=False, reward=0.0) as the final reward.
evaluate() is never called.

**Hypothesis**: The system prompt said "call final_step to submit" but didn't prohibit returning a text
response. The LLM generated a natural-language completion summary instead of a tool call.

**Intervention**: None needed — "Agent returned no actions" is unambiguous in the log. The step 022
obs confirmed tests passed.

**Fix** (`recipes/hello_swebench_verified.py` — `SWE_SYSTEM_PROMPT`):
Add "IMPORTANT: You MUST call the `final_step` tool to submit — do NOT just write a text response.
Every turn must end with a tool call."

**Follow-up candidate** (cube-harness framework): `episode.py:213` silently discards work when the
agent returns no actions. When `task.accept_agent_stop=True`, the framework should call evaluate()
rather than using the stale 0.0 reward. This would correctly surface reward=1.0 when the agent's
fix was right but it forgot to call `final_step`.

**Result**: Run 6 — `psf__requests-1142` reward=1.0 in 10 turns (agent correctly called final_step).
Combined run 6 result: **2/2 tasks solved** (first 100% solve rate across all runs).

## Iteration 11 — 2026-04-28 — get_records() always shows reward=0.0

**Tasks**: all (harness-level bug affecting every benchmark)

**What was broken**: `ExperimentResult.get_records()` returned `reward=0.0` for every completed
episode, masking all successes. Root cause: `get_exp_record()` read `reward` from `summary_stats`
(which stores the field as `final_reward`, not `reward`). The authoritative value is in
`reward_info.reward`. Same bug affected `cost_usd` (stored as `cost` in `summary_stats`).

**Fix** (`src/cube_harness/results.py`):
Explicitly pull `reward` from `meta.reward_info["reward"]`; remap `cost` → `cost_usd`. All
existing records (including earlier episode.metadata.json files) now report correctly.

**Result**: Both flask-5014 and requests-1142 from run 082247 correctly report reward=1.0 after fix.

## Iteration 12 — 2026-04-28 — Ray workers crash with working_dir=None and missing packages

**Tasks**: all (Ray parallelism blocked)

**What was broken**: Two bugs in `exp_runner.py`:
1. `runtime_env={"working_dir": None, ...}` — Ray's uv hook raises `TypeError: path_or_uri must
   be a string, got <class 'NoneType'>` when `working_dir` is explicitly set to `None`.
2. Ray workers couldn't import `swebench_verified_cube` or other editable/PEP-723 packages because
   workers don't inherit the calling process's `sys.path` — they use their own bare Python env.

**Fix** (`src/cube_harness/exp_runner.py`):
Drop `working_dir` key from `runtime_env` entirely. Propagate `sys.path` to workers via
`PYTHONPATH` env var so editable installs and PEP-723 isolated venvs are visible to workers.
Also fix recipe: rename `episode_timeout` → `step_timeout_s` to match actual `run_with_ray` signature.

**Result**: 20-task parallel run completed. 5/5 concurrency limited by Daytona free tier (10 GiB
total). Reduced `n_cpus` to 5. Overall 13/16 completed tasks passed (81% pass rate).

## Iteration 13 — 2026-04-28 — pytest-dev runner, _apply_patch logging, prompt clarity

**Tasks**: `pytest-dev__pytest-5809`, `pallets__flask-5014`

**pytest-5809**: `_build_test_cmd` added `--no-header` for all non-django/sympy repos. But old pytest
versions inside the testbed (the repo being tested) don't support this flag, causing eval to fail
with "unrecognized arguments: --no-header" regardless of fix quality.

**flask-5014**: Agent ran existing test suite, saw all tests pass, concluded task was done — never
implemented the required change (ValueError for empty Blueprint name). The system prompt said
"verify your fix by running tests" but did not say "the existing tests pass before your fix too."

**_apply_patch**: All three patching methods (git apply, git apply --reject, patch) can fail silently.
The function returned without error, evaluation found no test function, and the real cause was hidden
behind "test not found" messages.

**Fix** (`cubes/swebench-verified-cube/src/swebench_verified_cube/task.py`):
- Add `pytest-dev` case to `_build_test_cmd`: use `python -m pytest -rN -p no:cacheprovider` (no `--no-header`)
- Log a warning in `_apply_patch` when all three methods fail

**Fix** (`recipes/hello_swebench_verified.py`):
Add explicit note: "The existing test suite will pass before your fix. Do NOT call final_step just
because existing tests pass. Only call final_step after you have modified the source code."

**Result**: Pending verification run on pytest-5809 and flask-5014.
