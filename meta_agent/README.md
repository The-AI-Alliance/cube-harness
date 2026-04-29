# Meta-Agent

Systematic debugging and improvement of the agent stack through iterative
eval-analyse-fix loops. The meta-agent inspects failures, diagnoses root causes
across the full stack (benchmarks, tools, BrowserGym, WorkArena, agent scaffolding),
and applies targeted fixes.

## Structure

```
meta_agent/
├── journal/         # Per-session logs (1 file per debugging session)
├── recipes/         # Benchmark-specific experiment configs
└── README.md        # This file
```

**Hints and task precision** live in each cube's source tree (e.g.
`cubes/workarena/src/workarena_cube/agent_hints.py`) since they are imported
by the benchmark code at runtime.

## Skill

The meta-agent Claude Code skill is at `.claude/commands/meta-agent.md`.
Invoke with `/meta-agent`.
