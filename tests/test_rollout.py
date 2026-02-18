"""Integration tests for agentlab2.rl.rollout module."""

import json
from pathlib import Path

from litellm import Message

from agentlab2.agent import Agent, AgentConfig
from agentlab2.core import Action, ActionSchema, AgentOutput, Observation
from agentlab2.environment import EnvConfig
from agentlab2.llm import LLMCall, LLMConfig, Prompt, Usage
from agentlab2.rl.rollout import RolloutResult, rollout

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "rollout_result.json"

_APRIEL_LLM_CONFIG = LLMConfig(model_name="hosted_vllm/ServiceNow-AI/Apriel-1.6-15b-Thinker")


def _make_tool_call(name: str, arguments: str, tc_id: str) -> dict:
    """Build a tool_call dict for a Message."""
    return {"id": tc_id, "type": "function", "function": {"name": name, "arguments": arguments}}


# Scripted turns for the deterministic agent.
# Each entry: (actions, llm_call_kwargs)
_SCRIPT: list[tuple[list[Action], dict]] = [
    # Turn 0: click a button
    (
        [Action(id="call_0", name="click", arguments={"element_id": "btn_1"})],
        {
            "id": "llm_0",
            "timestamp": "2025-01-01T00:00:00",
            "prompt": Prompt(messages=[{"role": "user", "content": "Click the button."}]),
            "output": Message(
                content=None,
                role="assistant",
                tool_calls=[_make_tool_call("click", '{"element_id": "btn_1"}', "call_0")],
            ),
            "usage": Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        },
    ),
    # Turn 1: type text into an input
    (
        [Action(id="call_1", name="type_text", arguments={"element_id": "input_1", "text": "hello"})],
        {
            "id": "llm_1",
            "timestamp": "2025-01-01T00:00:01",
            "prompt": Prompt(
                messages=[
                    {"role": "user", "content": "Click the button."},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [_make_tool_call("click", '{"element_id": "btn_1"}', "call_0")],
                    },
                    {"role": "tool", "content": "Clicked on btn_1", "tool_call_id": "call_0"},
                    {"role": "user", "content": "Now type hello."},
                ],
            ),
            "output": Message(
                content="I'll type hello.",
                role="assistant",
                tool_calls=[_make_tool_call("type_text", '{"element_id": "input_1", "text": "hello"}', "call_1")],
            ),
            "usage": Usage(prompt_tokens=30, completion_tokens=10, total_tokens=40),
        },
    ),
    # Turn 2: two actions in one step (click + type)
    (
        [
            Action(id="call_2a", name="click", arguments={"element_id": "btn_2"}),
            Action(id="call_2b", name="type_text", arguments={"element_id": "input_2", "text": "world"}),
        ],
        {
            "id": "llm_2",
            "timestamp": "2025-01-01T00:00:02",
            "prompt": Prompt(messages=[{"role": "user", "content": "Click btn_2 and type world."}]),
            "output": Message(
                content="I'll do both.",
                role="assistant",
                tool_calls=[
                    _make_tool_call("click", '{"element_id": "btn_2"}', "call_2a"),
                    _make_tool_call("type_text", '{"element_id": "input_2", "text": "world"}', "call_2b"),
                ],
            ),
            "usage": Usage(prompt_tokens=40, completion_tokens=15, total_tokens=55),
        },
    ),
    # Turn 3: finish
    (
        [Action(name="final_step", arguments={})],
        {
            "id": "llm_3",
            "timestamp": "2025-01-01T00:00:03",
            "prompt": Prompt(messages=[{"role": "user", "content": "Finish."}]),
            "output": Message(content="Done.", role="assistant"),
            "usage": Usage(prompt_tokens=50, completion_tokens=5, total_tokens=55),
        },
    ),
]


class _DeterministicAgentConfig(AgentConfig):
    """Agent config that creates a deterministic 4-turn agent with scripted LLM calls."""

    def make(self, action_set: list[ActionSchema] | None = None, **kwargs) -> "_DeterministicAgent":
        return _DeterministicAgent(config=self)


class _DeterministicAgent(Agent):
    """Agent that produces fixed actions and LLM calls for exactly 4 turns.

    Turn 0: clicks "btn_1" (tool_calls only, no content)
    Turn 1: types "hello" into "input_1" (content + tool_calls, with history)
    Turn 2: clicks "btn_2" and types "world" (two actions, content + tool_calls)
    Turn 3: calls final_step (content only)
    """

    name = "DeterministicAgent"
    description = "Scripted agent for testing"
    input_content_types = ["text"]
    output_content_types = ["action"]

    def __init__(self, config: _DeterministicAgentConfig) -> None:
        super().__init__(config)
        self._turn = 0

    def step(self, obs: Observation) -> AgentOutput:
        turn = self._turn
        self._turn += 1
        actions, llm_kwargs = _SCRIPT[turn]
        llm_call = LLMCall(llm_config=_APRIEL_LLM_CONFIG, **llm_kwargs)
        return AgentOutput(actions=actions, llm_calls=[llm_call])


def _result_to_dict(result: RolloutResult) -> dict:
    """Serialize RolloutResult to a comparable dict."""
    return {
        "text_pairs": [
            {
                "prompt_text": tp.prompt_text,
                "response_text": tp.response_text,
                "reward": tp.reward,
                "images": tp.images,
            }
            for tp in result.text_pairs
        ],
        "reward": result.reward,
    }


def test_rollout_matches_stored_fixture(mock_tool_config, mock_task) -> None:
    """Run a deterministic rollout and verify results match the stored golden fixture.

    The deterministic agent runs 4 turns:
      - Turn 0: clicks btn_1 (tool_calls only) → env reward 0.0
      - Turn 1: types "hello" into input_1 (content + tool_calls + history) → env reward 0.0
      - Turn 2: clicks btn_2 and types "world" (two actions) → env reward 0.0
      - Turn 3: final_step (content only) → env validates → reward 1.0, done
    """
    agent_config = _DeterministicAgentConfig()
    env_config = EnvConfig(task=mock_task, tool_config=mock_tool_config)

    result = rollout(agent_config=agent_config, env_config=env_config, max_steps=10)

    result_dict = _result_to_dict(result)
    expected = json.loads(FIXTURE_PATH.read_text())
    assert result_dict == expected
