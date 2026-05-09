# Meta-Agent

Systematic debugging and improvement of the agent stack through iterative
eval-analyse-fix loops. The meta-agent inspects failures, diagnoses root causes
across the full stack (benchmarks, tools, BrowserGym, WorkArena, agent scaffolding),
and applies targeted fixes.

## Structure

```
meta_agent/
├── recipes/         # Benchmark-specific experiment configs
└── README.md        # This file
```

**Hints and task precision** live in each cube's source tree (e.g.
`cubes/workarena/src/workarena_cube/agent_hints.py`) since they are imported
by the benchmark code at runtime.

**Per-session journals** live in `~/cube_meta_agent_journal/` (machine-local,
not committed). One markdown file per debugging session; convention and
template are in the skill at `.claude/commands/meta-agent.md`.

## Skill

The meta-agent Claude Code skill is at `.claude/commands/meta-agent.md`.
Invoke with `/meta-agent`.
