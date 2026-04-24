# Proposal: Cube CI — PR smoke tests (cube-harness) + nightly monitoring (cube-registry)

## Scope of this document

This document covers two related but separately owned systems:

1. **PR-level smoke tests** — owned by **cube-harness** (implemented here)
2. **Nightly monitoring matrix + dashboard** — owned by **cube-registry** (design only; implementation in cube-registry openspec)

---

## Part 1 — PR smoke tests (cube-harness)

### What

Run `cube test <name>` (debug suite, scripted agent, no LLM) for every cube in `cubes/` on every PR. Fast and cheap.

**Key invariant:** debug agents are always scripted/oracle — never LLM-based. No LLM API keys in CI, ever. Only infra secrets (HF token, Docker daemon).

### Tier structure for PR CI

| Tier | Trigger | Cubes | Infra needed |
|------|---------|-------|--------------|
| 1 | Every PR, no secrets | arithmetic, miniwob, terminalbench, swebench-* | GH runner (Playwright, Docker) |
| 1+ | Every PR, with secret | workarena | `HUGGING_FACE_HUB_TOKEN` |

Jobs skip gracefully when their required secret is absent (e.g. external fork PRs).

### Implementation (Phase 1, in this PR)

`.github/workflows/cube-ci-fast.yml` — MiniWob (no secret) + WorkArena (HF token).
Remaining tier-1 cubes added incrementally as they stabilize.

---

## Part 2 — Nightly monitoring matrix (cube-registry)

### Vision

A nightly job runs `cube test <name>` for every cube registered in cube-registry, across every supported infra target, and publishes results to a monitoring dashboard. This replaces the ad-hoc "did it work?" check with structured, historical data.

### Matrix: cube × infra

Rows = cubes, columns = infra targets. Cells show pass/fail + key stats. Sparse matrix is fine — not every cube makes sense on every infra.

```
              | GH runner | Docker | AWS | Azure | HF-hosted |
miniwob       |    ✓      |        |     |       |           |
arithmetic    |    ✓      |        |     |       |           |
terminalbench |    ✓      |   ✓    |     |       |           |
workarena     |           |        |     |       |     ✓     |
swebench-*    |    ✓      |   ✓    |     |       |           |
osworld       |           |        |  ✓  |   ✓   |           |
webarena-*    |           |        |  ✓  |   ✓   |           |
```

### Statistics collected per cell

- Pass / fail (with auto-retry, default 2× configurable)
- L1 provisioning time (package install)
- L2 provisioning time (benchmark setup)
- L3 provisioning time (per-task reset)
- Total episode wall-clock time

### Journal

Every run appends to a persistent journal (structured log). Enables:
- Tracking failure trends over time
- Detecting flakiness (fail on 1 of 2 retries)
- Comparing provisioning times across infra and cube versions

### Dashboard

A page showing the matrix as a table. Cells are clickable → full logs for that cube × infra × run. Latest run status visible on each cube's registry page.

### What runs nightly

1. All cubes in the registry pinned to their published PyPI version.
2. All cubes in `cube-harness/cubes/` on their current `main` branch, **if** the local version differs from the published PyPI version (version check before running).

### Cube metadata needed (Phase 2 — requires cube-standard change)

To make the matrix declarative rather than hardcoded, `BenchmarkMetadata` gains:

```python
class CIConfig(TypedBaseModel):
    tier: int                          # 1=every-PR  2=nightly  3=weekly
    supported_infra: list[str] = []    # e.g. ["gh-runner", "docker", "aws"]
    required_secrets: list[str] = []   # env var names (non-standard secrets only)
    service_containers: list[str] = [] # docker-compose services
```

`cube test --list --tier=1` emits the tier-1 cube list so fast-CI workflows can be generated rather than hardcoded.

Each registered cube declares its required secrets. Common infra secrets (AWS_ACCESS_KEY_ID, AZURE_SUBSCRIPTION_ID) are pre-configured in cube-registry. Cube-specific secrets (e.g. HUGGING_FACE_HUB_TOKEN for WorkArena's hosted SN pool) are registered alongside the cube submission.

### Ownership

The nightly job, journal storage, dashboard, and registry integration are implemented and maintained in **cube-registry**. This document is a design reference; the authoritative openspec delta lives there.
