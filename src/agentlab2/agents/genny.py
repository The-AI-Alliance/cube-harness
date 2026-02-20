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
from typing import Protocol

from litellm import Message
from termcolor import colored

from agentlab2.agent import Agent, AgentConfig
from agentlab2.core import Action, ActionSchema, AgentOutput, Observation
from agentlab2.llm import LLM, LLMCall, LLMConfig, Prompt

logger = logging.getLogger(__name__)


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

_DEFAULT_SUMMARIZE_VERBOSE_SYSTEM_PROMPT = """\
You are a summarization assistant for an AI agent.
Given a goal, prior summaries, and the latest observation, produce a structured summary."""

_DEFAULT_SUMMARIZE_COT_SYSTEM_PROMPT = """\
You are a reasoning assistant for an AI agent.
Given a goal, prior summaries, and the latest observation, briefly reason about what happened and what it means."""

_DEFAULT_SUMMARIZE_VERBOSE_PROMPT = """\
Summarize the latest observation concisely. Include:
- What was observed (key changes, current state, errors)
- Progress toward the goal

Then add a '## Key Facts' section with durable facts worth preserving across compactions."""

_DEFAULT_SUMMARIZE_COT_PROMPT = """\
In 2-3 sentences, reason about the latest observation: what happened, what it means for the goal, and what to do next."""


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


def _truncate_message(msg: dict, max_chars: int) -> dict:
    content = msg.get("content", "")
    if isinstance(content, str) and len(content) > max_chars:
        return {**msg, "content": content[:max_chars] + "… [truncated]"}
    return msg


# ---------------------------------------------------------------------------
# ToolAdapter — isolates text vs. native tool interface
# ---------------------------------------------------------------------------


class ToolAdapter(Protocol):
    def encode(self, tools: list[ActionSchema], messages: list[dict | Message]) -> tuple[list[dict], list[dict | Message]]:
        """Return (api_tools, api_messages). api_tools is empty when baked into messages."""
        ...

    def decode(self, response: Message) -> list[Action]:
        """Extract actions from LLM response."""
        ...


class NativeToolAdapter:
    """Passes tools natively via the LLM API tool_use interface."""

    def encode(self, tools: list[ActionSchema], messages: list[dict | Message]) -> tuple[list[dict], list[dict | Message]]:
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

    def encode(self, tools: list[ActionSchema], messages: list[dict | Message]) -> tuple[list[dict], list[dict | Message]]:
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


class GennyAgentConfig(AgentConfig):
    # Core
    llm_config: LLMConfig
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT
    react_prompt: str = _DEFAULT_REACT_PROMPT

    # Tool interface
    tools_as_text: bool = False  # True = fn sigs in system prompt + <tool_call> parsing

    # Summarize pass
    enable_summarize: bool = False  # False = skip summarize pass entirely
    summarize_cot_only: bool = False  # True = concise CoT; False = verbose + Key Facts
    summarize_llm_config: LLMConfig | None = None  # None = reuse llm_config
    summarize_verbose_system_prompt: str = _DEFAULT_SUMMARIZE_VERBOSE_SYSTEM_PROMPT
    summarize_cot_system_prompt: str = _DEFAULT_SUMMARIZE_COT_SYSTEM_PROMPT
    summarize_verbose_prompt: str = _DEFAULT_SUMMARIZE_VERBOSE_PROMPT
    summarize_cot_prompt: str = _DEFAULT_SUMMARIZE_COT_PROMPT

    # Observation window
    render_last_n_obs: int | None = None  # None = all

    # Compaction
    max_history_tokens: int | None = None  # None = disabled; drops old groups when exceeded
    max_summaries_tokens: int | None = None  # None = disabled; LLM-compresses old summaries

    # Misc
    max_obs_chars: int | None = None  # None = no truncation
    max_actions: int | None = None  # None = unlimited

    def make(self, action_set: list[ActionSchema] | None = None, **kwargs) -> "GennyAgent":
        return GennyAgent(config=self, action_schemas=action_set or [])


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class GennyAgent(Agent):
    name: str = "genny"
    description: str = "Genny — phase 1 context management: summarize pass, windowed history, tool adapters."
    input_content_types: list[str] = ["image/png", "image/jpeg", "text/plain", "application/json"]
    output_content_types: list[str] = ["application/json"]

    def __init__(self, config: GennyAgentConfig, action_schemas: list[ActionSchema]):
        self.config = config
        self.llm: LLM = config.llm_config.make()
        self.summarize_llm: LLM = (config.summarize_llm_config or config.llm_config).make()
        self.token_counter = config.llm_config.make_counter()
        self.action_schemas: list[ActionSchema] = action_schemas
        self.tool_adapter: ToolAdapter = TextToolAdapter() if config.tools_as_text else NativeToolAdapter()
        self.goal: list[dict] = []
        self.summaries: list[str] = []
        self.history: list[list[dict | Message]] = []  # groups: one per obs or asst turn
        self._actions_cnt: int = 0

    def step(self, obs: Observation) -> AgentOutput:
        if self.config.max_actions is not None and self._actions_cnt >= self.config.max_actions:
            logger.info("Max actions reached, returning empty action.")
            return AgentOutput(actions=[])

        obs_messages = self._obs_to_messages(obs)
        self._ingest_obs(obs_messages)

        llm_calls: list[LLMCall] = []
        if self.config.enable_summarize:
            summary, sum_call = self._summarize_pass(obs_messages)
            self.summaries.append(summary)
            llm_calls.append(sum_call)
            self._maybe_compress_summaries()

        self._maybe_compact_history()

        response, act_call = self._act_pass()
        llm_calls.append(act_call)
        self.history.append([response])
        self._actions_cnt += 1
        return AgentOutput(actions=self.tool_adapter.decode(response), llm_calls=llm_calls)

    def _obs_to_messages(self, obs: Observation) -> list[dict]:
        messages = obs.to_llm_messages()
        if self.config.max_obs_chars is not None:
            messages = [_truncate_message(m, self.config.max_obs_chars) for m in messages]
        return messages

    def _ingest_obs(self, obs_messages: list[dict]) -> None:
        """On step 0 extract goal; on subsequent steps append obs group to history."""
        if not self.goal:
            self.goal = [obs_messages[0]]
            if len(obs_messages) > 1:
                self.history.append(obs_messages[1:])
        else:
            self.history.append(obs_messages)

    def _summarize_pass(self, obs_messages: list[dict]) -> tuple[str, LLMCall]:
        """Call summarize LLM on goal + prior summaries + latest obs."""
        if self.config.summarize_cot_only:
            system = self.config.summarize_cot_system_prompt
            user_prompt = self.config.summarize_cot_prompt
        else:
            system = self.config.summarize_verbose_system_prompt
            user_prompt = self.config.summarize_verbose_prompt
        prior_summary_msgs = [{"role": "assistant", "content": s} for s in self.summaries]
        messages: list[dict | Message] = [
            {"role": "system", "content": system},
            *self.goal,
            *prior_summary_msgs,
            *obs_messages,
            {"role": "user", "content": user_prompt},
        ]
        prompt = Prompt(messages=messages)
        response = self.summarize_llm(prompt)
        llm_config = self.config.summarize_llm_config or self.config.llm_config
        llm_call = LLMCall(llm_config=llm_config, prompt=prompt, output=response.message, usage=response.usage)
        return response.message.content or "", llm_call

    def _maybe_compress_summaries(self) -> None:
        if self.config.max_summaries_tokens is None or len(self.summaries) < 2:
            return
        summary_msgs = [{"role": "assistant", "content": s} for s in self.summaries]
        if self.token_counter(messages=summary_msgs) > self.config.max_summaries_tokens:
            self._compress_summaries()

    def _compress_summaries(self) -> None:
        """LLM-merge the first half of summaries into one to stay within token budget."""
        midpoint = max(1, len(self.summaries) // 2)
        combined = "\n\n---\n\n".join(self.summaries[:midpoint])
        messages: list[dict] = [
            {"role": "system", "content": "Compress these summaries into a single concise summary preserving all key facts."},
            {"role": "user", "content": combined},
        ]
        prompt = Prompt(messages=messages)
        response = self.summarize_llm(prompt)
        compressed = response.message.content or combined
        logger.info(f"Compressed {midpoint} summaries into one.")
        self.summaries = [compressed, *self.summaries[midpoint:]]

    def _maybe_compact_history(self) -> None:
        """Drop old history groups when raw history exceeds max_history_tokens."""
        if self.config.max_history_tokens is None:
            return
        flat = [msg for group in self.history for msg in group]
        if self.token_counter(messages=flat) > self.config.max_history_tokens:
            self._drop_old_history()

    def _drop_old_history(self) -> None:
        """Remove the first half of history groups (covered by summaries)."""
        midpoint = len(self.history) // 2
        if midpoint % 2 == 1:
            midpoint += 1  # keep group boundaries aligned (obs/asst pairs)
        if midpoint == 0 or midpoint >= len(self.history):
            return
        logger.info(f"Dropping {midpoint} history groups (covered by summaries).")
        self.history = self.history[midpoint:]

    def _act_pass(self) -> tuple[Message, LLMCall]:
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
        llm_call = LLMCall(llm_config=self.config.llm_config, prompt=prompt, output=response.message, usage=response.usage)
        return response.message, llm_call

    def _choose_context(self) -> list[dict | Message]:
        """Assemble the full context window for the act pass."""
        messages: list[dict | Message] = [{"role": "system", "content": self.config.system_prompt}]
        messages.extend(self.goal)
        for summary in self.summaries:
            messages.append({"role": "assistant", "content": summary})
        messages.extend(self._windowed_history())
        messages.append({"role": "user", "content": self.config.react_prompt})
        return messages

    def _windowed_history(self) -> list[dict | Message]:
        """Return flattened history groups, limited to last render_last_n_obs observations."""
        n = self.config.render_last_n_obs
        if n is None:
            groups = self.history
        else:
            tail_len = n * 2  # n obs groups + n asst groups
            groups = self.history[-tail_len:] if tail_len < len(self.history) else self.history
        return [msg for group in groups for msg in group]
