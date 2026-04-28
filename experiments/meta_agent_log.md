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
