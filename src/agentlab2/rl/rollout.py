import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

from agentlab2.agent import AgentConfig
from agentlab2.core import AgentOutput, EnvironmentOutput, Trajectory
from agentlab2.environment import EnvConfig
from agentlab2.episode import Episode
from agentlab2.llm import LLMCall
from agentlab2.rl.llm_call_renderer import TextPair, llm_call_to_text_pair

logger = logging.getLogger(__name__)


@dataclass
class RolloutMetrics:
    n_llm_calls: int
    n_step_errors: int
    n_observations: int
    n_steps: int
    total_execution_time: float
    agent_execution_time: float
    environment_execution_time: float
    env_step_time: float
    agent_step_time: float


@dataclass
class RolloutResult:
    """Result of a single rollout containing training data for RL."""

    text_pairs: list[TextPair]
    reward: float
    metrics: RolloutMetrics


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

    reward = trajectory.reward_info.get("reward", 0.0)
    text_pairs = []
    calls_with_rewards = _extract_llm_calls(trajectory)
    for call, step_reward in calls_with_rewards:
        text_pairs.append(llm_call_to_text_pair(call, step_reward))

    metrics = _compute_metrics(trajectory)
    return RolloutResult(text_pairs=text_pairs, reward=reward, metrics=metrics)


def _compute_metrics(trajectory: Trajectory) -> RolloutMetrics:
    """Compute rollout metrics from a trajectory."""
    n_llm_calls = 0
    n_step_errors = 0
    n_observations = 0
    n_steps = 0
    agent_execution_time = 0.0
    environment_execution_time = 0.0

    for step in trajectory.steps:
        step_duration = (step.end_time or 0.0) - (step.start_time or 0.0)
        if isinstance(step.output, AgentOutput):
            n_steps += 1
            n_llm_calls += len(step.output.llm_calls)
            agent_execution_time += step_duration
            if step.output.error is not None:
                n_step_errors += 1
        elif isinstance(step.output, EnvironmentOutput):
            n_observations += 1
            environment_execution_time += step_duration
            if step.output.error is not None:
                n_step_errors += 1

    total_execution_time = (trajectory.end_time or 0.0) - (trajectory.start_time or 0.0)

    return RolloutMetrics(
        n_llm_calls=n_llm_calls,
        n_step_errors=n_step_errors,
        n_observations=n_observations,
        n_steps=n_steps,
        total_execution_time=total_execution_time,
        agent_execution_time=agent_execution_time,
        environment_execution_time=environment_execution_time,
        env_step_time=environment_execution_time / n_observations if n_observations > 0 else 0.0,
        agent_step_time=agent_execution_time / n_steps if n_steps > 0 else 0.0,
    )


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
