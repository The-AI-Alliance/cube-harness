"""MiniAgent — faithful port of mini-SWE-agent (github.com/SWE-agent/mini-swe-agent).

Design:
- Flat message list (system + instance + repeated assistant/tool pairs). No rolling
  summaries, no windowed obs, no compression. Same as upstream default.py.
- Single bash tool via native LLM tool calls (OpenAI tool-call format via LiteLLM).
- Submission detected by magic string "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" as the
  first non-whitespace line of bash stdout with a zero exit code.
- Prompts are verbatim copies from upstream swebench.yaml.

Key difference from upstream:
- Submission detection happens at the START of the next step (when obs arrives)
  rather than immediately after exec. Functionally identical — one extra round-trip.
- NOTE: parallel_tool_calls=False (default). Upstream uses True. Enabling it may
  improve performance for tasks where the model issues exploratory commands in parallel.

Reference: src/minisweagent/agents/default.py, src/minisweagent/config/benchmarks/swebench.yaml
"""

import json
import logging

from cube.core import Action, ActionSchema, Observation
from cube.task import STOP_ACTION

from cube_harness.agent import Agent, AgentConfig
from cube_harness.core import AgentOutput, LLMCall
from cube_harness.llm import LLMConfig, Prompt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bash tool schema — verbatim from upstream actions_toolcall.py
# ---------------------------------------------------------------------------

BASH_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The bash command to execute"}},
            "required": ["command"],
        },
    },
}

# ---------------------------------------------------------------------------
# Prompts — verbatim from upstream swebench.yaml
# ---------------------------------------------------------------------------

_SYSTEM_TEMPLATE = "You are a helpful assistant that can interact with a computer shell to solve programming tasks."

_INSTANCE_TEMPLATE = """\
<pr_description>
Consider the following PR description:
{{task}}
</pr_description>

<instructions>
# Task Instructions

## Overview

You're a software engineer interacting continuously with a computer by submitting commands.
You'll be helping implement necessary changes to meet requirements in the PR description.
Your task is specifically to make changes to non-test files in the current directory in order to fix the issue described in the PR description in a way that is general and consistent with the codebase.
<IMPORTANT>This is an interactive process where you will think and issue AT LEAST ONE command, see the result, then think and issue your next command(s).</important>

For each response:

1. Include a THOUGHT section explaining your reasoning and what you're trying to accomplish
2. Provide one or more bash tool calls to execute

## Important Boundaries

- MODIFY: Regular source code files in /testbed (this is the working directory for all your subsequent commands)
- DO NOT MODIFY: Tests, configuration files (pyproject.toml, setup.cfg, etc.)

## Recommended Workflow

1. Analyze the codebase by finding and reading relevant files
2. Create a script to reproduce the issue
3. Edit the source code to resolve the issue
4. Verify your fix works by running your script again
5. Test edge cases to ensure your fix is robust

## Command Execution Rules

You are operating in an environment where

1. You issue at least one command
2. The system executes the command(s) in a subshell
3. You see the result(s)
4. You write your next command(s)

Each response should include:

1. **Reasoning text** where you explain your analysis and plan
2. At least one tool call with your command

**CRITICAL REQUIREMENTS:**

- Your response SHOULD include reasoning text explaining what you're doing
- Your response MUST include AT LEAST ONE bash tool call. You can make MULTIPLE tool calls in a single response when the commands are independent (e.g., searching multiple files, reading different parts of the codebase).
- Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
- However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment variables from files

Example of a CORRECT response:
<example_response>
I need to understand the Builder-related code. Let me find relevant files and check the project structure.

[Makes multiple bash tool calls: {"command": "ls -la"}, {"command": "find src -name '*.java' | grep -i builder"}, {"command": "cat README.md | head -50"}]
</example_response>

## Environment Details

- You have a full Linux shell environment
- Always use non-interactive flags (-y, -f) for commands
- Avoid interactive tools like vi, nano, or any that require user input
- You can use bash commands or invoke any tool that is available in the environment
- You can also create new tools or scripts to help you with the task
- If a tool isn't available, you can also install it

## Submission

When you've completed your work, you MUST submit your changes as a git patch.
Follow these steps IN ORDER, with SEPARATE commands:

Step 1: Create the patch file
Run `git diff -- path/to/file1 path/to/file2 > patch.txt` listing only the source files you modified.
Do NOT commit your changes.

<IMPORTANT>
The patch must only contain changes to the specific source files you modified to fix the issue.
Do not submit file creations or changes to any of the following files:

- test and reproduction files
- helper scripts, tests, or tools that you created
- installation, build, packaging, configuration, or setup scripts unless they are directly part of the issue you were fixing (you can assume that the environment is already set up for your client)
- binary or compiled files
</IMPORTANT>

Step 2: Verify your patch
Inspect patch.txt to confirm it only contains your intended changes and headers show `--- a/` and `+++ b/` paths.

Step 3: Submit (EXACT command required)
You MUST use this EXACT command to submit:

```bash
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt
```

If the command fails (nonzero exit status), it will not submit.

<CRITICAL>
- Creating/viewing the patch and submitting it MUST be separate commands (not combined with &&).
- If you modify patch.txt after verifying, you SHOULD verify again before submitting.
- You CANNOT continue working (reading, editing, testing) in any way on this task after submitting.
</CRITICAL>
</instructions>"""


# ---------------------------------------------------------------------------
# Observation rendering — verbatim logic from upstream observation_template
# ---------------------------------------------------------------------------

_TRUNCATION_WARNING = """\
<warning>
The output of your last command was too long.
Please try a different command that produces less output.
If you're looking at a file you can try use head, tail or sed to view a smaller number of lines selectively.
If you're using grep or find and it produced too much output, you can use a more selective search pattern.
If you really need to see something from the full command's output, you can redirect output to a file and then search in that file.
</warning>"""


def _render_observation(
    obs_text: str,
    max_chars: int = 10_000,
    head_chars: int = 5_000,
    tail_chars: int = 5_000,
) -> str:
    if len(obs_text) < max_chars:
        return f"<output>\n{obs_text}\n</output>"
    elided = len(obs_text) - max_chars
    return (
        f"{_TRUNCATION_WARNING}\n"
        f"<output_head>\n{obs_text[:head_chars]}\n</output_head>\n"
        f"<elided_chars>\n{elided} characters elided\n</elided_chars>\n"
        f"<output_tail>\n{obs_text[-tail_chars:]}\n</output_tail>"
    )


def _is_submission(obs_text: str) -> bool:
    """Return True if obs_text's first non-whitespace line is the magic submission string."""
    lines = obs_text.lstrip().splitlines()
    return bool(lines) and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


# ---------------------------------------------------------------------------
# MiniAgentConfig / MiniAgent
# ---------------------------------------------------------------------------


class MiniAgentConfig(AgentConfig):
    """Configuration for MiniAgent — mirrors upstream swebench.yaml defaults."""

    llm_config: LLMConfig
    step_limit: int = 250
    obs_max_chars: int = 10_000
    obs_head_chars: int = 5_000
    obs_tail_chars: int = 5_000
    system_prompt: str = _SYSTEM_TEMPLATE
    instance_template: str = _INSTANCE_TEMPLATE

    @property
    def agent_name(self) -> str:
        return f"MiniAgent-{self.llm_config.model_name}".replace("/", "_")

    def make(self, action_set: list[ActionSchema] | None = None, **kwargs: object) -> "MiniAgent":
        return MiniAgent(self)


class MiniAgent(Agent):
    """Minimal bash-only agent: flat message list, native tool calls, magic-string exit."""

    name = "mini_agent"
    description = "Faithful port of mini-SWE-agent: flat history, single bash tool, magic-string submission."
    input_content_types = ["text/plain"]
    output_content_types = ["application/json"]

    def __init__(self, config: MiniAgentConfig) -> None:
        self.config = config
        self._llm = config.llm_config.make()
        self._messages: list[dict | object] = []
        self._pending_tool_call_id: str | None = None
        self._turn: int = 0

    def step(self, obs: Observation) -> AgentOutput:
        obs_messages = obs.to_llm_messages()
        obs_text: str = obs_messages[0].get("content", "") if obs_messages else ""

        if not self._messages:
            # First call: obs_text is the problem statement — populate message list.
            rendered_instance = self.config.instance_template.replace("{{task}}", obs_text)
            self._messages = [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": rendered_instance},
            ]
        else:
            # Subsequent calls: obs_text is bash stdout from the previous action.
            if _is_submission(obs_text):
                logger.info("MiniAgent: submission magic string detected — stopping episode")
                return AgentOutput(actions=[Action(id="stop", name=STOP_ACTION.name, arguments={})])
            rendered = _render_observation(
                obs_text,
                self.config.obs_max_chars,
                self.config.obs_head_chars,
                self.config.obs_tail_chars,
            )
            self._messages.append({"role": "tool", "tool_call_id": self._pending_tool_call_id, "content": rendered})

        if self._turn >= self.config.step_limit:
            logger.info("MiniAgent: step limit %d reached — stopping", self.config.step_limit)
            return AgentOutput(actions=[Action(id="stop", name=STOP_ACTION.name, arguments={})])

        prompt = Prompt(messages=self._messages, tools=[BASH_TOOL])
        llm_response = self._llm(prompt)
        self._messages.append(llm_response.message)
        self._turn += 1

        llm_call = LLMCall(
            tag="act",
            llm_config=self.config.llm_config,
            prompt=prompt,
            output=llm_response.message,
            usage=llm_response.usage,
        )

        # Capture any reasoning text the model included alongside tool calls.
        thoughts: str | None = getattr(llm_response.message, "content", None) or None

        tool_calls = getattr(llm_response.message, "tool_calls", None) or []
        if not tool_calls:
            logger.warning("MiniAgent: no tool calls returned at turn %d — stopping", self._turn)
            return AgentOutput(
                actions=[Action(id="stop", name=STOP_ACTION.name, arguments={})],
                llm_calls=[llm_call],
                thoughts=thoughts,
            )

        # NOTE: Only the first tool call is used (parallel_tool_calls=False in LLMConfig).
        # Upstream mini-SWE-agent uses parallel_tool_calls=True; enabling it here may improve
        # performance for tasks that benefit from parallel exploration commands.
        tc = tool_calls[0]
        self._pending_tool_call_id = tc.id
        args = tc.function.arguments
        if isinstance(args, str):
            args = json.loads(args)
        cmd: str = args.get("command", "")

        return AgentOutput(
            actions=[Action(name="bash", arguments={"command": cmd})],
            llm_calls=[llm_call],
            thoughts=thoughts,
        )
