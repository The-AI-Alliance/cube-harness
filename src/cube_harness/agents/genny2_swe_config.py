"""Shared Genny2 agent configuration for SWE-bench and TerminalBench recipes."""

from cube_harness.agents.genny2 import BudgetConfig, Genny2Config
from cube_harness.llm import LLMConfig

SYSTEM_PROMPT = "You are a helpful assistant that can interact with a computer shell to solve programming tasks. If an action seems to have no apparent effect, avoid retrying it."

_WORKFLOW_BLOCK = """\
Suggested approach:
1. Reproduce: confirm the current behavior — run the relevant test or command to observe the issue.
2. Explore: read the relevant source files directly with `cat`, `grep`, or `find` to locate the root cause.
3. Fix: apply the minimal change needed to resolve the issue.
4. Verify: run the relevant test or command to confirm the fix works and nothing else broke. If it does not, make a focused adjustment and try again.\
"""

_WORKFLOW_BLOCK_GENERIC = """\
Suggested approach:
1. Explore: inspect the environment — read relevant files, run diagnostic commands, or reproduce the issue before making changes.
2. Act: execute the minimal steps needed to complete the task.
3. Verify: confirm the result meets the requirement. If it does not, make a focused adjustment and try again.\
"""

_WORKFLOW_BLOCK_TBENCH = """\
Suggested approach:
1. Explore: read the task, inspect /app and any relevant files to understand the environment.
2. Implement: write, modify, or install what is needed to satisfy the task.
3. Verify: run a quick sanity check to confirm your solution works.
4. Finish: call final_step when you are done.\
"""

INSTANCE_TEMPLATES: dict[str, str] = {
    "minimal": "{{task}}",
    "workflow": f"{{{{task}}}}\n\n{_WORKFLOW_BLOCK}",
    "workflow-generic": f"{{{{task}}}}\n\n{_WORKFLOW_BLOCK_GENERIC}",
    "workflow-tbench": f"{{{{task}}}}\n\n{_WORKFLOW_BLOCK_TBENCH}",
}

# Production default: generic 3-step workflow, works across SWE-bench, TerminalBench, and similar.
DEFAULT_TEMPLATE = "workflow-generic"

MODEL_CONFIGS: dict[str, LLMConfig] = {
    "gpt-5.4-mini": LLMConfig(
        model_name="azure/gpt-5.4-mini",
        temperature=1.0,
        tool_choice="required",
        parallel_tool_calls=True,
    ),
    "gpt-5.4": LLMConfig(
        model_name="azure/gpt-5.4",
        temperature=1.0,
        tool_choice="required",
        parallel_tool_calls=True,
    ),
    "haiku": LLMConfig(
        model_name="anthropic/claude-haiku-4-5",
        temperature=0.0,
        tool_choice="required",
        parallel_tool_calls=False,
        set_cache_control="auto",
    ),
    "sonnet": LLMConfig(
        model_name="anthropic/claude-sonnet-4-6",
        temperature=0.0,
        tool_choice="required",
        parallel_tool_calls=True,
        set_cache_control="auto",
    ),
}


def make_agent_config(
    model_key: str,
    template: str = DEFAULT_TEMPLATE,
    max_actions: int = 150,
    cost_limit: float = 1.0,
) -> Genny2Config:
    return Genny2Config(
        llm_config=MODEL_CONFIGS[model_key],
        system_prompt=SYSTEM_PROMPT,
        goal_template=INSTANCE_TEMPLATES[template],
        flat_history=True,
        step_prompt="",
        max_format_errors=3,
        budget=BudgetConfig(max_actions=max_actions, cost_limit=cost_limit),
    )
