"""Genny agent — Phase 1: context management.

Context layout per act call:
    system_prompt          (static)
    [tool definitions]     (if tools_as_text=True, injected into system by TextToolAdapter)
    goal                   (static, extracted from step 0)
    summaries[-k:]         (rolling compressed history, one string per summarize pass)
    history[-n obs:]       (windowed raw obs/asst groups)
    react_prompt           (static instruction)
"""

import json
import logging
import re
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Protocol, cast

from litellm import Message
from termcolor import colored

from agentlab2.agent import Agent, AgentConfig
from agentlab2.core import Action, ActionSchema, AgentOutput, Observation
from agentlab2.environment import STOP_ACTION
from agentlab2.llm import LLM, LLMCall, LLMConfig, Prompt

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
Be concise and focused."""

_DEFAULT_REACT_PROMPT = """\
Review the latest observation and produce the next action.
Think step by step:
1. What does the observation show?
2. What was the effect of the last action?
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
    """Format action schemas as Python-style function signatures for text-mode injection."""
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


def _format_summaries_block(summaries: list[str]) -> str:
    """Combine past-step summaries into a single assistant message with step headers."""
    parts = ["## Summary of past interactions"]
    for i, summary in enumerate(summaries, 1):
        parts.append(f"### Step {i}\n\n{summary}")
    return "\n\n".join(parts)


def _obs_section_header(n: int | None) -> str:
    """User-message header that precedes the windowed observation history."""
    if n is None:
        return "## Most recent observations"
    return f"## {n} most recent observations"


def _format_action_list(actions: "list[Action]") -> str:
    """Format a list of actions as a compact text string."""
    parts = [f"{a.name}({', '.join(f'{k}={v!r}' for k, v in a.arguments.items())})" for a in actions]
    return ", ".join(parts) if parts else "no action"


def _extract_act_summary(response: "Message", actions: "list[Action]") -> str:
    """Build a step summary from an act-pass response for use as a rolling COT entry.

    Combines the LLM's reasoning text (extended thinking or inline content) with a
    formatted description of the action(s) taken. Used when enable_summarize=False so
    that prior reasoning is visible in future steps via self.summaries.
    """
    cot = getattr(response, "reasoning_content", None) or (response.content or "")
    parts = []
    if cot:
        parts.append(cot.strip())
    parts.append(f"Action: {_format_action_list(actions)}")
    return "\n\n".join(parts)


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

    # Tool interface
    tools_as_text: bool = False  # True = fn sigs in system prompt + <tool_call> parsing

    # Summarize pass
    enable_summarize: bool = False  # False = extract COT from act pass; True = separate summarize LLM call
    summarize_cot_only: bool = False  # True = concise CoT; False = verbose + Key Facts
    summarize_llm_config: LLMConfig | None = None  # None = reuse llm_config
    summarize_verbose_prompt: str = _DEFAULT_SUMMARIZE_VERBOSE_PROMPT
    summarize_cot_prompt: str = _DEFAULT_SUMMARIZE_COT_PROMPT

    # Observation window
    render_last_n_obs: int | None = None  # None = all

    # Misc
    max_obs_chars: int | None = None  # None = no truncation
    max_actions: int | None = None  # None = unlimited

    def make(self, action_set: list[ActionSchema] | None = None, **kwargs) -> "Genny":
        return Genny(config=self, action_schemas=action_set or [])


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class Genny(Agent):
    name: str = "genny"
    description: str = "Genny — phase 1 context management: summarize pass, windowed history, tool adapters."
    input_content_types: list[str] = ["image/png", "image/jpeg", "text/plain", "application/json"]
    output_content_types: list[str] = ["application/json"]

    def __init__(self, config: GennyConfig, action_schemas: list[ActionSchema]):
        self.config = config
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
        self.summaries: list[str] = []
        self.history: list[list[dict | Message]] = []  # groups: one per obs or asst turn
        self._actions_cnt: int = 0

    def step(self, obs: Observation) -> AgentOutput:
        if self.config.max_actions is not None and self._actions_cnt >= self.config.max_actions:
            logger.info("Max actions reached, issuing STOP action.")
            return AgentOutput(actions=[Action(name=STOP_ACTION.name, arguments={})])

        profiler = Profiler()
        other_llm_calls: dict[str, list[LLMCall]] = {}

        with profiler("context"):
            obs_messages = self._obs_to_messages(obs)
            self._ingest_obs(obs_messages)

        rationale: str | None = None
        if self.config.enable_summarize:
            # Summarize the current obs (already in history via _ingest_obs).
            with profiler("summarize"):
                summary, sum_call = self._summarize_past()
            rationale = summary  # capture before action is appended below
            self.summaries.append(summary)
            other_llm_calls["summary"] = [sum_call]

        with profiler("act"):
            response, act_call = self._act()
        actions = self.tool_adapter.decode(response)

        if self.config.enable_summarize:
            # Append the decided action to the current step's summary so the
            # summaries block alternates reasoning → action, matching the COT mode format.
            self.summaries[-1] += f"\n\nAction: {_format_action_list(actions)}"
        else:
            # No explicit summarize LLM — extract COT from the act response for rolling context.
            rationale = getattr(response, "reasoning_content", None) or response.content or None
            step_summary = _extract_act_summary(response, actions)
            if step_summary:
                self.summaries.append(step_summary)

        asst_group: list[dict | Message] = [response]
        self.history.append(asst_group)
        self._actions_cnt += 1
        return AgentOutput(
            actions=actions,
            llm_calls=[act_call],
            other_llm_calls=other_llm_calls,
            profiling=profiler.data,
            action_rationale=rationale or None,
        )

    def _obs_to_messages(self, obs: Observation) -> list[dict | Message]:
        messages = cast(list[dict | Message], obs.to_llm_messages())
        if self.config.max_obs_chars is not None:
            messages = cast(list[dict | Message], [_truncate_message(m, self.config.max_obs_chars) for m in messages])
        return messages

    def _ingest_obs(self, obs_messages: list[dict | Message]) -> None:
        """On step 0 extract goal; on subsequent steps append obs group to history."""
        if not self.goal:
            self.goal = [obs_messages[0]]
            if len(obs_messages) > 1:
                self.history.append(obs_messages[1:])
        else:
            self.history.append(obs_messages)

    def _build_base_prompt(self, exclude_last_summary: bool = False) -> list[dict | Message]:
        """Build the shared prompt prefix used by both _summarize_past and _choose_context.

        Both passes extend this prefix with their specific final instruction, ensuring the
        [system, goal, summaries_block, obs_header, windowed_history] prefix is byte-for-byte
        identical → prompt-cache hit on the entire prefix.

        When exclude_last_summary=True the last entry in self.summaries is omitted from the
        collapsed block so it can be placed after the obs (used by _choose_context when
        enable_summarize=True).
        """
        messages: list[dict | Message] = [{"role": "system", "content": self.config.system_prompt}]
        messages.extend(self.goal)
        past_summaries = self.summaries[:-1] if (exclude_last_summary and self.summaries) else list(self.summaries)
        if past_summaries:
            messages.append({"role": "assistant", "content": _format_summaries_block(past_summaries)})
        windowed = self._windowed_history()
        if windowed:
            messages.append({"role": "user", "content": _obs_section_header(self.config.render_last_n_obs)})
            messages.extend(windowed)
        return messages

    def _summarize_past(self) -> tuple[str, LLMCall]:
        """Extend the shared base prompt with the summarize instruction.

        Uses _build_base_prompt() + tool_adapter.encode() so the system prompt
        transformation and tool definitions are byte-for-byte identical to the act pass
        → prompt-cache hit on the full shared prefix (system, tools, goal, summaries, obs).
        The summarize LLM has tool_choice="none" so it responds with text, not tool calls.
        """
        user_prompt = (
            self.config.summarize_cot_prompt if self.config.summarize_cot_only else self.config.summarize_verbose_prompt
        )
        messages = self._build_base_prompt()
        messages.append({"role": "user", "content": user_prompt})
        api_tools, api_messages = self.tool_adapter.encode(self.action_schemas, messages)
        prompt = Prompt(messages=api_messages, tools=api_tools)
        response = self.summarize_llm(prompt)
        llm_call = LLMCall(
            llm_config=self._summarize_llm_config, prompt=prompt, output=response.message, usage=response.usage
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
            llm_config=self.config.llm_config, prompt=prompt, output=response.message, usage=response.usage
        )
        return response.message, llm_call

    def _choose_context(self) -> list[dict | Message]:
        """Extend the shared base prompt with the act instruction.

        When enable_summarize=True, self.summaries[-1] is the current step's reasoning
        (from _summarize_past). It is excluded from the collapsed block and placed *after*
        the obs window so the LLM sees obs → reasoning → act_prompt.

        When enable_summarize=False, all summaries (COT extracted from prior act passes)
        go into the collapsed block; react_prompt instructs the LLM to reason inline.
        """
        messages = self._build_base_prompt(exclude_last_summary=self.config.enable_summarize)
        if self.config.enable_summarize and self.summaries:
            messages.append({"role": "assistant", "content": self.summaries[-1]})
        final_prompt = self.config.act_prompt if self.config.enable_summarize else self.config.react_prompt
        messages.append({"role": "user", "content": final_prompt})
        return messages

    def _windowed_history(self) -> list[dict | Message]:
        """Return flattened history groups, limited to last render_last_n_obs observations.

        When render_last_n_obs is set, only obs groups are included (not their preceding asst
        groups). Leading tool-role messages are stripped from each obs group so the prompt stays
        structurally valid — the paired tool_calls live in the dropped asst groups, but those
        actions are already captured in self.summaries, making the tool results redundant.
        """
        if self.config.render_last_n_obs is None:
            return [msg for group in self.history for msg in group]
        n = self.config.render_last_n_obs
        # Obs groups start with role 'user'/'tool'; asst groups start with role 'assistant'.
        obs_groups = [
            g
            for g in self.history
            if g and (g[0].role if isinstance(g[0], Message) else g[0].get("role", "")) != "assistant"
        ]
        selected = obs_groups[-n:] if n < len(obs_groups) else obs_groups
        result: list[dict | Message] = []
        for group in selected:
            # Drop leading tool-role messages (their paired tool_calls are in dropped asst groups)
            start = next(
                (i for i, m in enumerate(group) if not (isinstance(m, dict) and m.get("role") == "tool")), len(group)
            )
            result.extend(group[start:])
        return result
