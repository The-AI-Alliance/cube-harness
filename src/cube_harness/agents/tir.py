import json
import logging
from typing import Literal

from cube.core import Action, ActionSchema, Observation

from cube_harness.agent import Agent, AgentConfig
from cube_harness.core import AgentOutput, LLMCall
from cube_harness.llm import LLMConfig, Prompt
from cube_harness.utils import parse_actions

logger = logging.getLogger(__name__)


class TirAgentConfig(AgentConfig):
    """TIR-style tool-calling agent.

    Prompt shape per step:
    - optional system prompt
    - accumulated conversation history:
      - initial user task observation
      - assistant tool-call messages
      - tool result messages from the environment

    No extra React instruction message is injected each turn.
    """

    llm_config: LLMConfig
    system_prompt: str = ""
    max_actions: int = 3
    # Fallback behavior when model emits plain text without explicit tool calls.
    # - off: never synthesize MathAnswer
    # - boxed_only: synthesize MathAnswer only when a \boxed{...} answer is present
    # - always: synthesize MathAnswer with full assistant content
    mathanswer_fallback: Literal["off", "boxed_only", "always"] = "off"

    def make(self, action_set: list[ActionSchema]) -> "TirAgent":
        return TirAgent(config=self, tools=action_set)


class TirAgent(Agent):
    name: str = "tir_agent"
    description: str = "A minimal TIR-style agent that iterates over tool calls."
    input_content_types: list[str] = ["image/png", "image/jpeg", "text/plain", "application/json"]
    output_content_types: list[str] = ["application/json"]

    def __init__(self, config: TirAgentConfig, tools: list[ActionSchema]):
        self.config = config
        self.llm = config.llm_config.make()
        self.tools: list[dict] = [tool.as_dict() for tool in tools]

        self.history: list[dict] = []
        self._actions_cnt = 0

    def max_actions_reached(self) -> bool:
        return self._actions_cnt >= self.config.max_actions

    def _build_prompt_messages(self) -> list[dict]:
        messages: list[dict] = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.extend(self.history)
        return messages

    def _supports_mathanswer(self) -> bool:
        return any(tool.get("function", {}).get("name") == "MathAnswer" for tool in self.tools)

    @staticmethod
    def _extract_last_boxed(content: str) -> str | None:
        start = content.rfind("\\boxed{")
        if start < 0:
            return None

        i = start + len("\\boxed{")
        depth = 1
        while i < len(content):
            ch = content[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return content[start : i + 1]
            i += 1
        return None

    def _fallback_mathanswer_argument(self, content: str) -> str | None:
        mode = self.config.mathanswer_fallback
        stripped = content.strip()
        if not stripped or mode == "off":
            return None
        if mode == "always":
            return stripped
        if mode == "boxed_only":
            return self._extract_last_boxed(stripped)
        return None

    @staticmethod
    def _parse_actions_tolerant(llm_output) -> list[Action]:
        """Mirror native TIR tolerance for malformed tool-call arguments.

        Native TIR does not fail the rollout when tool-call arguments are malformed
        or not an object; it recovers to empty/default arguments.
        """
        try:
            return parse_actions(llm_output)
        except Exception:
            actions: list[Action] = []
            for tc in getattr(llm_output, "tool_calls", []) or []:
                name = getattr(getattr(tc, "function", None), "name", None)
                if not name:
                    continue
                raw_args = getattr(getattr(tc, "function", None), "arguments", {})
                if isinstance(raw_args, dict):
                    args = raw_args
                else:
                    try:
                        parsed = json.loads(raw_args) if isinstance(raw_args, str) else {}
                        args = parsed if isinstance(parsed, dict) else {}
                    except Exception:
                        args = {}
                actions.append(Action(id=getattr(tc, "id", None), name=name, arguments=args))
            return actions

    def step(self, obs: Observation) -> AgentOutput:
        if self.max_actions_reached():
            # Mirror TIR behavior: stop looping when max turns are reached,
            # without forcing a synthetic stop action.
            return AgentOutput(actions=[])

        self.history.extend(obs.to_llm_messages())
        prompt = Prompt(messages=self._build_prompt_messages(), tools=self.tools)

        llm_response = self.llm(prompt)
        llm_output = llm_response.message
        self.history.append(llm_output.model_dump(exclude_none=True) if hasattr(llm_output, "model_dump") else {
            "role": getattr(llm_output, "role", "assistant"),
            "content": getattr(llm_output, "content", ""),
        })
        self._actions_cnt += 1

        actions = self._parse_actions_tolerant(llm_output)
        # If the model emits plain text with no explicit tool call, optionally
        # recover a MathAnswer submission from the assistant text.
        if not actions and self._supports_mathanswer():
            content = (getattr(llm_output, "content", "") or "").strip()
            fallback_answer = self._fallback_mathanswer_argument(content)
            if fallback_answer:
                actions = [Action(name="MathAnswer", arguments={"answer": fallback_answer})]

        llm_call = LLMCall(
            llm_config=self.config.llm_config,
            prompt=prompt,
            prompt_tokens=llm_response.usage.prompt_tokens,
            output_tokens=llm_response.usage.completion_tokens,
            output=llm_output,
            usage=llm_response.usage,
            logprobs=llm_response.logprobs,
            completion_token_ids=llm_response.completion_token_ids,
            finish_reason=llm_response.finish_reason,
        )
        
        return AgentOutput(actions=actions, llm_calls=[llm_call])
