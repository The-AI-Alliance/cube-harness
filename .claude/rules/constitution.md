# The cube-harness Constitution

> **📝 To update this constitution**, use the Claude command `/update-constitution`.
> This ensures all dependent files are updated automatically.
> See `.claude/commands/update-constitution.md` for details.

**Version**: 0.1

**Mission**: Empower the open-source community with high-throughput data generation for training and standardized benchmark evaluation by creating a modular, scalable, and efficient agent research platform.

---

## Preamble: The "One Team" Mindset

We are scaling up to a distributed engineering effort. To prevent architectural entropy, every contributor agrees to uphold the following principles. We value **explicitness over magic**, **composition over inheritance**, and **protocols over implementations**.

---

## Pillar I: The Team Contract & Ownership

*How we organize, communicate, and own our work.*

### Explicit Ownership

**Directive**: Every file and feature has a clear owner. We distinguish between:

- **Horizontal Ownership (Infrastructure)**: Core capabilities used by everyone (e.g., Parallelism, Protocols).
- **Vertical Ownership (Features)**: End-to-end features (e.g., The WebAgent Benchmarks + Agents).

**Mechanism**: We'll assign ownership soon.

**The Rule**: If you need to modify a component, you must consult its Owner.

### The RFC Process

**Directive**: Any change that alters the Core API or affects multiple verticals requires a Request For Comments (RFC).

**Process**: Write a 1-page Google doc → Post to Team Channel → Tag the relevant Component Owners → Async Review → Decision.

---

## Pillar II: The Principle of Explicitness

*The codebase should be readable like a book. We reject "Magic" configuration and hidden states.*

### Python is the Configuration

We reject complex YAML hierarchies, opaque Hydra overrides, or massive bash scripts.

**Directive**: All configurations must be defined as strictly typed Python dataclasses.

**Tooling**: We use `tyro` for CLI generation.

**The Rule**: If you can't click "Go to Definition" in your IDE to see where a parameter comes from, it is forbidden.

### Composition Over Inheritance

We avoid deep inheritance trees where a subclass inherits 50 methods it doesn't use.

**Directive**: Build complex Agents/Benchmarks by nesting standard components ("Legos"), not by subclassing a "God Object."

**Pattern**:
- ❌ Bad: `class MyAgent(BaseAllKnowingAgent): ...`
- ✅ Good: `class MyAgent(Agent): def __init__(self, planner: Planner, memory: Memory): ...`

### No Global State

**Directive**: We do not use global variables, singletons, or module-level state that cannot be reset.

**The Test**: You must be able to instantiate two different Agents with two different configurations in the same Python process without them interfering with each other.

---

## Pillar III: The "Scalable Research" Philosophy

*We build for massive scale and efficiency, while maintaining a developer-friendly experience.*

### Local-Dev, Cloud-Scale

**Directive**: The system is designed for massive parallelism (Ray/Slurm) from day one, but the Agent Logic must remain debuggable.

**Mechanism**:
- **Agent Logic**: Must be testable on a single laptop process (Direct Mode).
- **Infrastructure**: Scale-specific features (e.g., distributed samplers) may require a cluster, but we provide local mocks where feasible.

**The Rule**: You should be able to `pdb` through an Agent's decision step on your laptop, even if you can't run the full on-policy training loop locally.

### The Inner Loop is Sacred (Efficiency)

**Directive**: The core Agent-Environment loop must be optimized for high-throughput on-policy sampling.

**Constraint**: Avoid blocking calls or heavy serialization in the critical path. The architecture must support asynchronous execution to maximize GPU/Environment utilization.

**Goal**: "samples per second" for training as a high priority. Features introducing overhead should be discussed.

### The "Escape Hatch" (Raw Access)

**Directive**: Abstractions must never prevent a user from inspecting the raw underlying object when necessary.

**Example**: A `BrowserTool` wrapper must expose the underlying Playwright `Page` object via a `.raw` or `.underlying` property for advanced researchers.

### Trace-First Engineering

**Directive**: Telemetry is not an afterthought. The "Trace" (logs, screenshots, tool outputs, reasoning steps) is a first-class data product. Profiling methodology should be thought at the core.

**Standard**: We adhere to the Agent Data Protocol (ADP). All agents must emit traces in this format.

---

## Pillar IV: The Protocol Strategy

*We define standards to play nice with the ecosystem (NeMo, LangChain, etc.).*

### Interfaces over Implementations

**Directive**: Core interactions (Agent ↔ Env) are defined via Protocols (Interfaces), not concrete classes.

**Goal**: This allows us to swap the backend (e.g., switching from a local Docker container to a NeMo Resource Server) without changing the Agent's code.

### Embrace Standards

**Directive**: We do not invent new standards if a working one exists.

- **LLM**: We use LiteLLM abstractions.
- **Tools**: We support the Model Context Protocol (MCP).
- **Data**: We use ADP.

### Hermetic Reproducibility

**Directive**: We value reproducibility.

**Requirement**: Every experiment run must capture:
- The exact git commit hash.
- The full Configuration object (dumped as YAML/JSON).
- The Docker container ID/hash of the environment.

---

## Pillar V: The Craft of Code

*We maintain a lean, high-quality codebase. Code is a liability, not an asset.*

### The Minimalist Imperative

**Directive**: We prefer a smaller, simpler codebase over one that supports every edge case. If a feature adds significant complexity but is rarely used, reject it.

**Action**: Refactoring to delete code is prioritized over adding non-critical features. If you can delete 100 lines by refactoring the Core, communicate with the team.

### Function Atomicity

**Directive**: Break long functions into logical sub-functions. A function should theoretically fit on a standard screen (approx. 50-80 lines).

**Goal**: Self-documenting code. We prefer named helpers like `_parse_observation()` over inline comments explaining a 50-line block.

### AI-Assisted, Human-Architected (No "Vibe Coding")

**Directive**: We use AI tools to generate snippets and search for solutions, but we never blindly paste large blocks of code ("Vibe Coding").

**Risk**: "Vibe coding" pollutes the codebase with verbose, hallucinated, or unoptimized logic.

**The Rule**: You must understand and curate every line you commit. If the AI wrote it, you must refactor and tighten it before merging.

### The Testing Pyramid

**Directive**: We prioritize high coverage with simple, fast unit tests. Extensive unit tests may slow down development refactoring.

**CI Rule**: The Core test suite must run fast (< 5 mins). Slow tests go to nightly builds.

**Style**: We enforce `black` formatting and static analysis (`ruff`/`mypy`) strictly.
