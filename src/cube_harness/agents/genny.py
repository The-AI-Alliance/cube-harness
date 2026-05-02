"""Genny agent — cache-friendly context management.

Two operating modes, both with a stable cacheable prefix:

Mode A — growing raw history (enable_summarize=False):
    system_prompt          (static)
    goal                   (static)
    hints / clarification  (static)
    [obs_0, asst_0, ...]   (completed rounds, grow by one pair each step)
    latest_obs             (the new observation, appended at context-build time)
    react_prompt           (static)

Mode B — rolling summaries (enable_summarize=True):
    system_prompt          (static)
    goal                   (static)
    hints / clarification  (static)
    asst: summary_1        (one message per step — NOT bundled)
    asst: summary_2
    ...
    asst: summary_k        ← rolling cache breakpoint lands here
    latest_obs             (shown to both sum and act passes)
    [asst: summary_{k+1}]  (act pass only, after summarize generates it)
    act_prompt / react_prompt

Summaries as separate messages (not a single bundled block) lets Anthropic's
prefix cache extend cleanly across steps: each step's cache ending at summary_k
is a valid prefix of the next step, which starts the same way and appends one more.
"""

import json
import logging
import re
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Protocol, cast

from cube.core import Action, ActionSchema, Observation
from cube.task import STOP_ACTION
from litellm import Message
from pydantic import Field
from termcolor import colored

from cube_harness.agent import Agent, AgentConfig
from cube_harness.core import AgentOutput
from cube_harness.llm import LLM, LLMCall, LLMConfig, Prompt

logger = logging.getLogger(__name__)


class Profiler:
    """Records named wall-clock spans; call as a context manager to record each span."""

    def __init__(self) -> None:
        self._data: dict[str, tuple[float, float]] = {}

    @contextmanager
    def __call__(self, name: str) -> Generator[None, None, None]:
        t_start = time.time()
        yield
        self._data[name] = (t_start, time.time())

    @property
    def data(self) -> dict[str, tuple[float, float]]:
        return self._data


# ---------------------------------------------------------------------------
# Default prompts
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM_PROMPT = """\
You are an expert AI agent. Understand the goal, take targeted actions, and reason clearly about progress.
Verify that each action had the intended effect before proceeding. Be concise and focused."""

_DEFAULT_REACT_PROMPT = """\
Review the latest observation and produce the next action.
Think step by step:
1. What does the observation show?
2. Did the last action have the intended effect? If the page state is unchanged or the action failed, do NOT repeat it — try a different element, method, or approach.
3. What is the best next action?
Then call the appropriate function."""

_DEFAULT_ACT_PROMPT = """\
Based on the reasoning above, call the appropriate function to perform the next action."""

_DEFAULT_SUMMARIZE_VERBOSE_PROMPT = """\
Summarize the latest observation concisely. Include:
- What was observed (key changes, current state, errors)
- Progress toward the goal

Then add a '## Key Facts' section with durable facts worth preserving across compactions.

Respond with text only — do not call any tools or functions."""

_DEFAULT_SUMMARIZE_COT_PROMPT = """\
In 2-3 sentences, reason about the latest observation: what happened, what it means for the goal, and what to do next.

Respond with text only — do not call any tools or functions."""


# ---------------------------------------------------------------------------
# Tool formatting helpers
# ---------------------------------------------------------------------------


def _json_type_to_python(json_type: str) -> str:
    return {
        "string": "str",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
        "array": "list",
        "object": "dict",
    }.get(json_type, "Any")


def _format_tools_as_text(tools: list[ActionSchema]) -> str:
    """Format action schemas as Python-style function signatures for text-mode injection.

    Used by TextToolAdapter for ablation studies vs. native tool calling. Parameters with
    complex JSON schemas (e.g. nested objects, $ref) render as 'Any' — best-effort display.
    If ablation shows no benefit over native tool calling, this adapter will be removed.
    """
    lines = ["## Available Functions"]
    for tool in tools:
        props = tool.parameters.get("properties", {})
        required = set(tool.parameters.get("required", []))
        args = []
        for pname, pinfo in props.items():
            ptype = _json_type_to_python(pinfo.get("type", "Any"))
            suffix = "" if pname in required else " = None"
            args.append(f"{pname}: {ptype}{suffix}")
        lines.append(f"def {tool.name}({', '.join(args)}) -> None:")
        if tool.description:
            lines.append(f'    """{tool.description}"""')
        lines.append("")
    lines += [
        "To call a function, respond with:",
        '<tool_call>{"name": "...", "arguments": {...}}</tool_call>',
    ]
    return "\n".join(lines)


def _format_action_list(actions: "list[Action]") -> str:
    """Format a list of actions as a compact text string."""
    parts = [f"{a.name}({', '.join(f'{k}={v!r}' for k, v in a.arguments.items())})" for a in actions]
    return ", ".join(parts) if parts else "no action"


def _get_reasoning(response: "Message") -> str:
    """Extract reasoning text from a response, checking all known fields across providers.

    Checks reasoning_content (OpenAI o-series / Anthropic streaming),
    then thinking_blocks (Anthropic extended thinking), then falls back to content.
    """
    if rc := getattr(response, "reasoning_content", None):
        return rc
    blocks = getattr(response, "thinking_blocks", None) or []
    block_text = " ".join(b.get("thinking", "") for b in blocks if isinstance(b, dict))
    if block_text:
        return block_text
    return response.content or ""


def _truncate_message(msg: dict, max_chars: int) -> dict:
    content = msg.get("content", "")
    if isinstance(content, str) and len(content) > max_chars:
        return {**msg, "content": content[:max_chars] + "… [truncated]"}
    return msg


# ---------------------------------------------------------------------------
# ToolAdapter — isolates text vs. native tool interface
# ---------------------------------------------------------------------------


class ToolAdapter(Protocol):
    def encode(
        self, tools: list[ActionSchema], messages: list[dict | Message]
    ) -> tuple[list[dict], list[dict | Message]]:
        """Return (api_tools, api_messages). api_tools is empty when baked into messages."""
        ...

    def decode(self, response: Message) -> list[Action]:
        """Extract actions from LLM response."""
        ...


class NativeToolAdapter:
    """Passes tools natively via the LLM API tool_use interface."""

    def encode(
        self, tools: list[ActionSchema], messages: list[dict | Message]
    ) -> tuple[list[dict], list[dict | Message]]:
        return [t.as_dict() for t in tools], messages

    def decode(self, response: Message) -> list[Action]:
        actions = []
        for tc in getattr(response, "tool_calls", None) or []:
            args = tc.function.arguments
            if isinstance(args, str):
                args = json.loads(args)
            if tc.function.name:
                actions.append(Action(id=tc.id, name=tc.function.name, arguments=args))
        return actions


class TextToolAdapter:
    """Injects function signatures into the system prompt; parses <tool_call> XML tags."""

    def encode(
        self, tools: list[ActionSchema], messages: list[dict | Message]
    ) -> tuple[list[dict], list[dict | Message]]:
        if not tools:
            return [], list(messages)
        sigs = _format_tools_as_text(tools)
        result: list[dict | Message] = list(messages)
        if result and isinstance(result[0], dict) and result[0].get("role") == "system":
            system_msg = dict(result[0])
            system_msg["content"] = system_msg["content"] + "\n\n" + sigs
            result[0] = system_msg
        else:
            result.insert(0, {"role": "system", "content": sigs})
        return [], result

    def decode(self, response: Message) -> list[Action]:
        content = getattr(response, "content", "") or ""
        actions = []
        for raw in re.findall(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL):
            try:
                data = json.loads(raw.strip())
                actions.append(Action(name=data["name"], arguments=data.get("arguments", {})))
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse tool_call: {raw!r} — {e}")
        return actions


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class GennyConfig(AgentConfig):
    # Core
    llm_config: LLMConfig
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT
    # react_prompt: reason-then-act, used when enable_summarize=False (COT embedded in act call)
    react_prompt: str = _DEFAULT_REACT_PROMPT
    # act_prompt: action-only, used when enable_summarize=True (reasoning done in summarize pass)
    act_prompt: str = _DEFAULT_ACT_PROMPT

    # Tool interface: False = native API tool_use (default); True = fn sigs in system prompt
    # + <tool_call> XML parsing (TextToolAdapter). Both modes are supported; tools_as_text
    # exists for ablation studies — if it shows no benefit it will be removed.
    tools_as_text: bool = False

    # Summarize pass
    enable_summarize: bool = False  # False = raw history mode; True = rolling summaries mode
    summarize_cot_only: bool = False  # True = concise CoT; False = verbose + Key Facts
    summarize_llm_config: LLMConfig | None = None  # None = reuse llm_config
    summarize_verbose_prompt: str = _DEFAULT_SUMMARIZE_VERBOSE_PROMPT
    summarize_cot_prompt: str = _DEFAULT_SUMMARIZE_COT_PROMPT

    # General hint injected after the goal in every step's context.
    # Use this when one hint applies to a whole task subset (one config per subset).
    hint: str = ""

    # Per-task hints: task_id -> hint text. Takes precedence over `hint` when a task_id match is found.
    # These are general or task-specific hints that help the LLM work better.
    task_hints: dict[str, str] = Field(default_factory=dict)

    # Per-task precision: task_id -> text that clarifies the goal when the task description
    # is under-defined (e.g. expected answer format, submission method). Injected as part of
    # the goal — not as a separate hint section.
    task_clarification: dict[str, str] = Field(default_factory=dict)

    # Misc
    max_obs_chars: int | None = None  # None = no truncation
    max_actions: int | None = None  # None = unlimited

    @property
    def agent_name(self) -> str:
        name = f"Genny-{self.llm_config.model_name}".replace("/", "_")
        if self.summarize_llm_config and self.summarize_llm_config.model_name != self.llm_config.model_name:
            name += f"+{self.summarize_llm_config.model_name}".replace("/", "_")
        return name

    def make(self, action_set: list[ActionSchema] | None = None, task_id: str | None = None, **kwargs) -> "Genny":
        return Genny(config=self, action_schemas=action_set or [], task_id=task_id)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class Genny(Agent):
    """ReAct-style agent with cache-friendly context management.

    Mode A (enable_summarize=False): raw history grows by one (obs, asst) pair per step.
    Latest obs is appended at context-build time, keeping the completed history as a stable
    cacheable prefix.

    Mode B (enable_summarize=True): a separate summarize LLM call produces a per-step
    summary stored as its own assistant message. Summaries accumulate as individual messages
    (not bundled) so each step's cache extends the previous step's. Latest obs is shown to
    both the summarize and act passes; the act pass also sees the new summary before the obs.
    """

    name: str = "genny"
    description: str = "Genny — phase 1 context management: summarize pass, windowed history, tool adapters."
    input_content_types: list[str] = ["image/png", "image/jpeg", "text/plain", "application/json"]
    output_content_types: list[str] = ["application/json"]

    def __init__(self, config: GennyConfig, action_schemas: list[ActionSchema], task_id: str | None = None):
        self.config = config
        self.task_id = task_id
        if task_id is None and (config.task_hints or config.task_clarification):
            logger.debug(
                "task_id is None — %d task_hints and %d task_clarifications not applied",
                len(config.task_hints),
                len(config.task_clarification),
            )
        # task_hints takes precedence over the general hint; falls back to hint if no match.
        self._task_hint: str = config.task_hints.get(task_id, config.hint) if task_id else config.hint
        # task_clarification is injected as part of the goal, not as a hint.
        self._task_clarification: str = config.task_clarification.get(task_id, "") if task_id else ""
        self.llm: LLM = config.llm_config.make()
        # Summarize LLM uses the same config as the act LLM (including tool_choice) so the
        # full request — messages, tools, and parameters — is identical between the two passes
        # → prompt-cache hit on the shared prefix. tool_choice is intentionally NOT overridden
        # to "none" because Azure/OpenAI include it in the cache key.
        self._summarize_llm_config = config.summarize_llm_config or config.llm_config
        self.summarize_llm: LLM = self._summarize_llm_config.make()
        self.token_counter = config.llm_config.make_counter()
        self.action_schemas: list[ActionSchema] = action_schemas
        self.tool_adapter: ToolAdapter = TextToolAdapter() if config.tools_as_text else NativeToolAdapter()
        self.goal: list[dict] = []
        self.summaries: list[str] = []  # Mode B: one entry per step
        self.history: list[list[dict | Message]] = []  # Mode A: completed (obs, asst) pairs
        self._latest_obs: list[dict | Message] = []  # current step's obs, not yet in history
        self._actions_cnt: int = 0

    def step(self, obs: Observation) -> AgentOutput:
        if self.config.max_actions is not None and self._actions_cnt >= self.config.max_actions:
            logger.info("Max actions reached, issuing STOP action.")
            return AgentOutput(actions=[Action(name=STOP_ACTION.name, arguments={})])

        profiler = Profiler()

        with profiler("context"):
            obs_messages = self._obs_to_messages(obs)
            self._ingest_obs(obs_messages)

        thoughts: str | None = None
        sum_call: LLMCall | None = None
        if self.config.enable_summarize:
            with profiler("summarize"):
                summary, sum_call = self._summarize_past()
            thoughts = summary
            self.summaries.append(summary)

        with profiler("act"):
            response, act_call = self._act()
        actions = self.tool_adapter.decode(response)

        if self.config.enable_summarize:
            self.summaries[-1] += f"\n\nAction: {_format_action_list(actions)}"
        else:
            thoughts = _get_reasoning(response) or None
            # Mode A: commit the completed (obs, asst) round to history.
            if self._latest_obs:
                self.history.append(self._latest_obs)
            self.history.append([response])

        llm_calls: list[LLMCall] = [act_call] + ([sum_call] if sum_call is not None else [])
        self._actions_cnt += 1
        return AgentOutput(
            actions=actions,
            llm_calls=llm_calls,
            profiling=profiler.data,
            thoughts=thoughts or None,
        )

    def _obs_to_messages(self, obs: Observation) -> list[dict | Message]:
        messages = cast(list[dict | Message], obs.to_llm_messages())
        if self.config.max_obs_chars is not None:
            messages = cast(list[dict | Message], [_truncate_message(m, self.config.max_obs_chars) for m in messages])
        return messages

    def _ingest_obs(self, obs_messages: list[dict | Message]) -> None:
        """On step 0 extract goal; on all steps park the obs in _latest_obs."""
        if not self.goal:
            self.goal = [obs_messages[0]]
            self._latest_obs = list(obs_messages[1:])
        else:
            self._latest_obs = list(obs_messages)

    def _build_base_prompt(self, exclude_last_summary: bool = False) -> list[dict | Message]:
        """Build the stable prompt prefix shared by summarize and act passes.

        Mode A: system + goal + hints + completed history (obs+asst pairs).
        Mode B: system + goal + hints + summaries as individual assistant messages.

        Latest obs is NOT included here — callers append it so the prefix up to the
        last summary/action remains byte-identical across both passes and across steps,
        enabling Anthropic's longest-prefix cache matching.
        """
        messages: list[dict | Message] = [{"role": "system", "content": self.config.system_prompt}]
        messages.extend(self.goal)
        if self._task_clarification:
            messages.append({"role": "user", "content": f"## Additional task details\n\n{self._task_clarification}"})
            messages.append({"role": "assistant", "content": "Understood."})
        if self._task_hint:
            messages.append({"role": "user", "content": f"## Task Hint\n\n{self._task_hint}"})
            messages.append({"role": "assistant", "content": "Understood, I'll keep this in mind."})
        if self.config.enable_summarize:
            summaries = self.summaries[:-1] if (exclude_last_summary and self.summaries) else self.summaries
            for s in summaries:
                messages.append({"role": "assistant", "content": s})
        else:
            for group in self.history:
                messages.extend(group)
        return messages

    def _summarize_past(self) -> tuple[str, LLMCall]:
        """Summarize the latest obs. Prompt: base_prefix + latest_obs + summarize_instruction.

        The base_prefix (system, goal, hints, prior summaries) is byte-for-byte identical
        to the act pass prefix → within-step cache hit between summarize and act.
        """
        user_prompt = (
            self.config.summarize_cot_prompt if self.config.summarize_cot_only else self.config.summarize_verbose_prompt
        )
        messages = self._build_base_prompt()
        messages.extend(self._latest_obs)
        messages.append({"role": "user", "content": user_prompt})
        api_tools, api_messages = self.tool_adapter.encode(self.action_schemas, messages)
        prompt = Prompt(messages=api_messages, tools=api_tools)
        response = self.summarize_llm(prompt)
        llm_call = LLMCall(
            tag="summary",
            llm_config=self._summarize_llm_config,
            prompt=prompt,
            output=response.message,
            usage=response.usage,
        )
        return response.message.content or "", llm_call

    def _act(self) -> tuple[Message, LLMCall]:
        """Build context, encode tools, call act LLM, return (response_message, llm_call)."""
        messages = self._choose_context()
        api_tools, api_messages = self.tool_adapter.encode(self.action_schemas, messages)
        prompt = Prompt(messages=api_messages, tools=api_tools)
        logger.info(f"Act pass — estimated prompt tokens: {self.token_counter(messages=api_messages)}")
        try:
            response = self.llm(prompt)
        except Exception as e:
            logger.exception(colored(f"LLM error in act pass: {e}", "red"))
            raise
        logger.info(
            f"LLM usage — prompt: {response.usage.prompt_tokens}, "
            f"completion: {response.usage.completion_tokens}, cost: ${response.usage.cost:.4f}"
        )
        llm_call = LLMCall(
            tag="act", llm_config=self.config.llm_config, prompt=prompt, output=response.message, usage=response.usage
        )
        return response.message, llm_call

    def _choose_context(self) -> list[dict | Message]:
        """Build the act-pass prompt.

        Mode B: base_prefix(exclude_last) + new_summary + latest_obs + act_prompt.
          Summaries are contiguous before the obs so the cache prefix ending at
          new_summary is a valid prefix of the next step's sum call.

        Mode A: base_prefix (completed history) + latest_obs + react_prompt.
          Step counter prepended to final user message only so preceding messages
          stay byte-for-byte identical across steps.
        """
        if self.config.enable_summarize:
            messages = self._build_base_prompt(exclude_last_summary=True)
            if self.summaries:
                messages.append({"role": "assistant", "content": self.summaries[-1]})
            messages.extend(self._latest_obs)
            final_prompt = self.config.act_prompt
        else:
            messages = self._build_base_prompt()
            messages.extend(self._latest_obs)
            final_prompt = self.config.react_prompt
        if self.config.max_actions is not None:
            final_prompt = f"[Step {self._actions_cnt + 1}/{self.config.max_actions}]\n\n{final_prompt}"
        messages.append({"role": "user", "content": final_prompt})
        return messages
