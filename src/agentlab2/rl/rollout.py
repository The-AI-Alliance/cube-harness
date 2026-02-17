import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

from agentlab2.agent import AgentConfig
from agentlab2.core import AgentOutput, EnvironmentOutput, Trajectory
from agentlab2.environment import EnvConfig
from agentlab2.episode import Episode
from agentlab2.llm import LLMCall
from agentlab2.rl.llm_call_renderer import llm_call_to_text_pair

logger = logging.getLogger(__name__)


@dataclass
class TextPair:
    """A single training sample from an LLM call."""

    prompt_text: str
    response_text: str
    reward: float | None = None


@dataclass
class RolloutResult:
    """Result of a single rollout containing training data for RL."""

    text_pairs: list[TextPair]
    reward: float


def rollout(
    agent_config: AgentConfig,
    env_config: EnvConfig,
    max_steps: int = 20,
) -> RolloutResult:
    """Run an agent on an environment and produce training data for RL.

    Executes a single episode, extracts all LLM calls from the trajectory,
    and converts them to training samples using the model-specific renderer.

    Args:
        agent_config: Configuration for the agent.
        env_config: Configuration for the environment (contains task inside).
        max_steps: Maximum number of agent turns.

    Returns:
        RolloutResult with training samples and episode reward.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        episode = Episode(
            id=0,
            output_dir=Path(tmpdir),
            agent_config=agent_config,
            env_config=env_config,
            max_steps=max_steps,
        )
        trajectory = episode.run()

    calls_with_rewards = _extract_llm_calls(trajectory)
    reward = trajectory.reward_info.get("reward", 0.0)
    text_pairs = []
    for call, step_reward in calls_with_rewards:
        prompt_text, response_text = llm_call_to_text_pair(call)
        text_pairs.append(TextPair(prompt_text=prompt_text, response_text=response_text, reward=step_reward))

    return RolloutResult(text_pairs=text_pairs, reward=reward)


def _extract_llm_calls(trajectory: Trajectory) -> list[tuple[LLMCall, float | None]]:
    """Extract all LLM calls from agent steps, paired with the reward from the following env step.

    Trajectory steps alternate: EnvironmentOutput, AgentOutput, EnvironmentOutput, ...
    For each AgentOutput at index i, the reward comes from the EnvironmentOutput at index i+1.
    All LLM calls within the same agent step share the same reward.
    """
    steps = trajectory.steps
    result: list[tuple[LLMCall, float | None]] = []
    for i, step in enumerate(steps):
        if not isinstance(step.output, AgentOutput):
            continue
        next_step = steps[i + 1] if i + 1 < len(steps) else None
        step_reward: float | None = None
        if next_step is not None and isinstance(next_step.output, EnvironmentOutput):
            step_reward = next_step.output.reward
        for call in step.output.llm_calls:
            result.append((call, step_reward))
    return result
