# Cubes

This folder contains concrete CUBE implementations — self-contained Python packages that wrap specific benchmarks (e.g., WebArena, SWE-bench, OSWorld) using the `cube` standard library.

Each cube implements the `Tool`, `Task`, and `Benchmark` interfaces from `cube-standard` (plus an optional `Container` config for sandboxed environments). Once wrapped, any cube can run in any CUBE-compatible harness.

**Building a new cube?** See the **[Authoring a CUBE guide](https://the-ai-alliance.github.io/cube-standard/authoring-a-cube)** in cube-standard — it covers the five-layer architecture, three starting paths (interactive `/new-cube` skill, copy an example, scaffold from the template), implementation order, and validation.
