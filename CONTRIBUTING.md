# Contributing to cube-harness

For contribution philosophy, DCO requirements, RFC process, and community guidelines, see the canonical [CONTRIBUTING.md in cube-standard](https://github.com/The-AI-Alliance/cube-standard/blob/main/CONTRIBUTING.md).

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

## Licenses

- **Code** — Apache 2.0 ([LICENSE.Apache-2.0](LICENSE.Apache-2.0))
- **Documentation** — CC BY 4.0 ([LICENSE.CC-BY-4.0](LICENSE.CC-BY-4.0))
- **Data** — CDLA Permissive 2.0 ([LICENSE.CDLA-2.0](LICENSE.CDLA-2.0))

## Community

- [GitHub Issues](https://github.com/The-AI-Alliance/cube-harness/issues) — bug reports and feature requests
- [GitHub Discussions](https://github.com/The-AI-Alliance/cube-harness/discussions) — design conversations and RFCs
- [Apply as a core contributor](https://forms.gle/JFiBi4ynfVLMghAH8) — if you want to help shape priorities

See also the AI Alliance [community repo](https://github.com/The-AI-Alliance/community/) for cross-project guidelines and the [Code of Conduct](https://github.com/The-AI-Alliance/community/blob/main/CODE_OF_CONDUCT.md).
