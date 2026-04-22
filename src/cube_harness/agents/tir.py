import logging

from cube.core import ActionSchema, Observation

from cube_harness.agent import Agent, AgentConfig
from cube_harness.core import AgentOutput, LLMCall
from cube_harness.llm import LLMConfig, Prompt, RLCollectorConfig
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

    llm_config: LLMConfig | RLCollectorConfig
    system_prompt: str = ""
    max_actions: int = 3

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

        llm_call = LLMCall(
            llm_config=self.config.llm_config,
            prompt=prompt,
            output=llm_output,
            usage=llm_response.usage,
            logprobs=llm_response.logprobs,
            completion_token_ids=llm_response.completion_token_ids,
            finish_reason=llm_response.finish_reason,
        )

        return AgentOutput(actions=parse_actions(llm_output), llm_calls=[llm_call])
