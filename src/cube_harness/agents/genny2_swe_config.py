"""Shared Genny2 agent configuration for SWE-bench recipes."""

from cube_harness.agents.genny2 import BudgetConfig, Genny2Config
from cube_harness.llm import LLMConfig

SYSTEM_PROMPT = "You are a helpful assistant that can interact with a computer shell to solve programming tasks. If an action seems to have no apparent effect, avoid retrying it."

_THOUGHT_BLOCK = "Before acting, take time to understand the task: read the relevant files, reproduce or confirm the issue, and identify the root cause before making changes."

_WORKFLOW_BLOCK = """\
Suggested approach:
1. Reproduce: confirm the current behavior — run the relevant test or command to observe the issue.
2. Explore: read the relevant source files directly with `cat`, `grep`, or `find` to locate the root cause.
3. Fix: apply the minimal change needed to resolve the issue.
4. Verify: run the relevant test or command to confirm the fix works and nothing else broke. If it does not, make a focused adjustment and try again.\
"""

INSTANCE_TEMPLATES: dict[str, str] = {
    "minimal": "{{task}}",
    "thought": f"{{{{task}}}}\n\n{_THOUGHT_BLOCK}",
    "workflow": f"{{{{task}}}}\n\n{_WORKFLOW_BLOCK}",
    "thought-workflow": f"{{{{task}}}}\n\n{_THOUGHT_BLOCK}\n\n{_WORKFLOW_BLOCK}",
}

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
    template: str = "thought-workflow",
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
