# cube-harness — Roadmap

> This roadmap reflects current priorities and is updated as the project evolves. For detailed proposals, open a GitHub Discussion or file an RFC via PR.

## Phase 1 — Alpha Stabilization (current)

Goal: stable harness, first wave of cubes running end-to-end.

- [x] Core loop: `Agent`, `Episode`, `Trajectory`, `Experiment`
- [x] Ray-based parallel execution (`ExpRunner`)
- [x] Gradio experiment viewer
- [x] Silverstream.ai Bench viewer integration
- [x] First cubes landing:
  - *Web agents:* MiniWob ✅, WebArena-Verified ✅ ([#214](https://github.com/The-AI-Alliance/cube-harness/pull/214)), WorkArena ✅
  - *Computer use (CUA):* OSWorld ✅
  - *SWE:* SWE-bench Verified + Live ✅, TerminalBench 2 ✅, LiveCodeBench ✅
- [ ] RL rollouts pipeline
- [ ] Stable `v0.1` API — freeze core interfaces, tag release
- [ ] Published documentation site

## Phase 2 — Platform Integrations & Cube Growth

Goal: integrate with major agent frameworks, grow to ~50 cubes.
- [ ] NemoGym integration — run cube-harness experiments from NemoGym
- [ ] AgentBeats integration — leaderboard and evaluation pipeline
- [ ] Other platform integrations** — ongoing discussions with framework maintainers
- [ ] 20-50 cubes** across broader categories
- [ ] Streaming observations — real-time delivery during long-horizon tasks
- [ ] Multi-agent episode support
- [ ] Interface with various agent standards

## Phase 3 — Broad Ecosystem

Goal: Scale CUBE to broader community.

> Phase 3 priorities will be shaped by what the community builds in Phase 2. Join the [discussions](https://github.com/The-AI-Alliance/cube-harness/discussions) to help define it.

## How to Influence the Roadmap

- Comment on existing [GitHub Issues](https://github.com/The-AI-Alliance/cube-harness/issues) or open a new one
- Start a [GitHub Discussion](https://github.com/The-AI-Alliance/cube-harness/discussions)
- Submit an RFC markdown draft via PR
- [Apply as a core contributor](https://forms.gle/JFiBi4ynfVLMghAH8) to help shape priorities directly
