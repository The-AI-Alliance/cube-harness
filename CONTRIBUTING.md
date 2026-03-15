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

All commits need a DCO sign-off: `git commit -s -m "..."`.

## Repo Layout

```
src/cube_harness/
  core/        # Agent, Episode, Trajectory, Experiment protocols
  runners/     # ExpRunner (sequential + Ray)
  viewer/      # Gradio experiment viewer
cubes/         # Built-in CUBE-standard benchmark wrappers
recipes/       # Example experiment scripts
tests/         # Test suite
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
