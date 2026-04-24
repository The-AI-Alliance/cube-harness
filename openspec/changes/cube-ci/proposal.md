# Proposal: Tiered Cube CI

## Problem

We have no automated way to smoke-test cube benchmarks in CI. The `cube test <name>` debug suite exists and works, but nothing runs it automatically. As the number of cubes grows, regressions in environment setup, task reset, or tool wiring go undetected until a human runs an experiment.

## Key invariant

**Debug agents are always scripted/oracle — never LLM-based.** This means cube CI requires only infra secrets (a ServiceNow instance, a Docker daemon, a cloud VM), never LLM API keys.

## Proposed tier structure

| Tier | When | Examples | Required infra |
|------|------|----------|----------------|
| 1 | Every PR | arithmetic, miniwob, terminalbench, swebench-* | GH runner only (Playwright, Docker) |
| 2 | Nightly | workarena | HF token or SNOW Docker image |
| 3 | Weekly | osworld, webarena-verified | Cloud VM (AWS/Azure) |

## What we're adding now (Phase 1)

Two GitHub Actions workflows:
- `cube-ci-fast.yml` — tier 1, every PR, starts with MiniWob
- `cube-ci-nightly.yml` — tier 2, daily cron, starts with WorkArena via `HUGGING_FACE_HUB_TOKEN`

## Phase 2 — metadata in cube-standard (requires upstream change)

Currently CI tier is implicit (hardcoded in workflow YAML). The long-term design is to make it declarative: each benchmark self-describes its CI requirements so the workflows can be generated or validated automatically.

### Proposed addition to `BenchmarkMetadata` (cube-standard)

```python
class CIConfig(TypedBaseModel):
    tier: int                          # 1=every-PR  2=nightly  3=weekly
    required_secrets: list[str] = []   # env var names that must be set
    service_containers: list[str] = [] # docker-compose services to spin up
```

`BenchmarkMetadata` gains an optional `ci: CIConfig | None = None` field.

Each cube's `debug.py` sets it on its benchmark class:
```python
class WorkArenaBenchmark(Benchmark):
    benchmark_metadata = BenchmarkMetadata(
        id="workarena",
        ci=CIConfig(tier=2, required_secrets=["HUGGING_FACE_HUB_TOKEN"]),
    )
```

`cube test` can then warn if required secrets are absent before attempting to run.

The GH workflow matrix can be generated from `cube test --list --tier=1` rather than being hardcoded.

## Out of scope

- Running full LLM-agent experiments in CI — that's the meta-agent initiative, separate concern.
- Benchmark result regression tracking — separate from smoke-test pass/fail.
