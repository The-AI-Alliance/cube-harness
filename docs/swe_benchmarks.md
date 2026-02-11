# SWE Benchmark Results

Comparison of AgentLab2 ReactAgent results on Terminal-Bench 2.0 against the public leaderboard.

## Terminal-Bench 2.0

**Benchmark**: 89 real-world terminal tasks (4 easy, 55 medium, 30 hard).
**Metric**: Accuracy (binary pass/fail per task, all pytest tests must pass).
**Agent**: AgentLab2 `ReactAgent` with `tool_choice="required"`, `max_actions=100`.
**Sandbox**: Daytona cloud sandboxes with task-specific Docker images.

### Results

| Agent | Model | Accuracy |
|-------|-------|----------|
| Codex CLI | GPT-5-Mini | 31.9% |
| **AgentLab2 ReactAgent** | **GPT-5-Mini** | **28.1%** |
| Terminus 2 | GPT-5-Mini | 24.0% |
| **AgentLab2 ReactAgent** | **GPT-5-Nano** | **12.4%** |
| Codex CLI | GPT-5-Nano | 11.5% |
| Terminus 2 | GPT-5-Nano | 7.9% |

### AgentLab2 Breakdown by Difficulty

| Model | Passed | Accuracy | Easy | Medium | Hard |
|-------|--------|----------|------|--------|------|
| GPT-5-Mini | 25/89 | 28.1% | 2/4 (50%) | 19/55 (34.5%) | 4/30 (13.3%) |
| GPT-5-Nano | 11/89 | 12.4% | 2/4 (50%) | 9/55 (16.4%) | 0/30 (0%) |

### Analysis

- **GPT-5-Mini (28.1%)**: Between Terminus 2 (24.0%) and Codex CLI (31.9%) with the same model, validating that the AgentLab2 infrastructure produces consistent results.
- **GPT-5-Nano (12.4%)**: Above Codex CLI (11.5%) with the same model, showing competitive performance for a general-purpose agent.
- The ReactAgent is a general-purpose agent without task-specific prompting. Specialized agents typically achieve higher scores with the same model.

### Reproducing

```bash
# Install dataset
uv run scripts/export_terminal_bench.py

# Run with GPT-5-Mini (full, 4 Ray workers)
uv run recipes/hello_tbench.py full --model openai/gpt-5-mini

# Run with GPT-5-Nano
uv run recipes/hello_tbench.py full --model openai/gpt-5-nano

# Debug mode (1 task, sequential)
uv run recipes/hello_tbench.py debug --model openai/gpt-5-mini
```

Results are saved to `~/agentlab_results/al2/tbench_*`.

---

*Leaderboard data from [tbench.ai](https://www.tbench.ai/leaderboard/terminal-bench/2.0) as of 2026-02-10.*
*AgentLab2 results from full runs (89/89 tasks) on 2026-02-11.*
