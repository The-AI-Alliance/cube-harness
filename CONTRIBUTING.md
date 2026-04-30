# Contributing to cube-harness

For contribution philosophy, DCO requirements, RFC process, and community guidelines, see the canonical [CONTRIBUTING.md in cube-standard](https://github.com/The-AI-Alliance/cube-standard/blob/main/CONTRIBUTING.md).

## OpenSpec — how we manage contracts

We follow the [OpenSpec](https://github.com/Fission-AI/OpenSpec) methodology. Each layer of
cube-harness has a living spec in `openspec/specs/<layer>/spec.md` that defines its public
API, invariants, and gotchas. Before modifying a layer, read its spec. After a PR that
changes a public contract, run `/update-openspec` in Claude Code to sync the spec.

For breaking changes, write a short delta proposal in `openspec/changes/<name>/` before
coding — this makes the contract change visible to the team before code lands.

Full workflow, delta format, and examples: [`openspec/README.md`](openspec/README.md).  
The methodology reference is in [cube-standard's openspec/README.md](https://github.com/The-AI-Alliance/cube-standard/blob/main/openspec/README.md).

## Setup

```bash
git clone https://github.com/The-AI-Alliance/cube-harness.git
cd cube-harness
make install           # uv sync --all-extras
pre-commit install --hook-type pre-commit --hook-type commit-msg
```

```bash
make lint    # ruff check + format (auto-fix)
make test    # pytest tests/
```

All commits need a [DCO sign-off](https://developercertificate.org/): `git commit -s -m "..."`. Running `make install` sets up a git hook that adds this automatically.

## Repo Layout

```
src/cube_harness/
  agent.py          # Agent protocol and AgentConfig base
  benchmark.py      # Benchmark interface for task collections
  core.py           # Data structures: Action, Observation, Trajectory, Task
  environment.py    # Environment and EnvConfig abstractions
  episode.py        # Episode execution and trajectory persistence
  experiment.py     # Experiment configuration and statistics
  exp_runner.py     # Sequential and Ray-based parallel execution
  llm.py            # LLM wrapper using LiteLLM
  storage.py        # Trajectory storage backends
  tool.py           # Tool abstraction for action spaces
  agents/           # Agent implementations (ReAct, Genny, …)
  tools/            # Tool implementations (Playwright, BrowserGym, …)
  benchmarks/       # Benchmark wrappers (MiniWob, WorkArena, …)
  metrics/          # Telemetry and tracing (OpenTelemetry-based)
  action_spaces/    # Browser action space protocols
  analyze/          # Trajectory analysis and XRay inspection utilities
  mcp/              # MCP server for exposing tools via Model Context Protocol
recipes/            # Example experiment scripts
tests/              # Test suite
```

## Adding a new cube

Cubes live under `cubes/<cube-name>/`. CI auto-discovers them — no workflow changes needed when adding a new cube.

Every cube must have a `Makefile` with these targets:

| Target | What it must do |
|--------|----------------|
| `make install` | `uv sync --all-extras && uv pip install -e .` plus any cube-specific setup (e.g. `uv run playwright install chromium`, metadata generation scripts) |
| `make test` | `uv run cube test <cube-name>` — runs the cube-standard debug suite. Must exit 0 on success. |

Optional but conventional:

| Target | What it does |
|--------|-------------|
| `make unit-test` | `uv run pytest -n auto tests/` — if the cube has fast unit tests ||

**Secret dependencies:** If `make test` requires a secret (e.g. `HUGGING_FACE_HUB_TOKEN`), handle it in the Makefile — skip gracefully or fail with a clear message. The CI workflow passes secrets through as env vars but is otherwise fully generic. Never add cube-specific logic to the workflow file.

**Testing layers:**
- `make test` (this repo, pre-merge) — verifies the cube works correctly using its own debug suite. No LLM, no harness stack involved.
- cube-registry nightly — verifies the published cube version works against each registered cloud provider. Owned by cube-registry, not this repo.

## Licenses

- **Code** — Apache 2.0 ([LICENSE.Apache-2.0](LICENSE.Apache-2.0))
- **Documentation** — CC BY 4.0 ([LICENSE.CC-BY-4.0](LICENSE.CC-BY-4.0))
- **Data** — CDLA Permissive 2.0 ([LICENSE.CDLA-2.0](LICENSE.CDLA-2.0))

## Community

- [GitHub Issues](https://github.com/The-AI-Alliance/cube-harness/issues) — bug reports and feature requests
- [GitHub Discussions](https://github.com/The-AI-Alliance/cube-harness/discussions) — design conversations and RFCs
- [Apply as a core contributor](https://forms.gle/JFiBi4ynfVLMghAH8) — if you want to help shape priorities

See also the AI Alliance [community repo](https://github.com/The-AI-Alliance/community/) for cross-project guidelines and the [Code of Conduct](https://github.com/The-AI-Alliance/community/blob/main/CODE_OF_CONDUCT.md).
