"""Test script to verify error handling and experiment resumption features."""

import logging
import tempfile
from pathlib import Path
from typing import Protocol

from agentlab2.agent import Agent, AgentConfig
from agentlab2.benchmark import Benchmark
from agentlab2.core import Action, ActionSchema, AgentOutput, Observation, Task
from agentlab2.environment import EnvConfig
from agentlab2.episode import Episode
from agentlab2.experiment import Experiment
from agentlab2.tool import Tool, ToolConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Reduce verbosity of episode logging
logging.getLogger("agentlab2.episode").setLevel(logging.WARNING)


# --- Mock Tool ---
class MockActionSpace(Protocol):
    """Simple action space for testing."""

    def noop(self) -> str:
        """No operation."""
        ...


class MockTool(Tool):
    """Mock tool for testing."""

    action_space = MockActionSpace

    def __init__(self):
        self.step_count = 0

    def noop(self) -> str:
        """No operation action for testing."""
        self.step_count += 1
        return "noop executed"


class MockToolConfig(ToolConfig):
    """Mock tool configuration."""

    def make(self) -> MockTool:
        return MockTool()


# --- Mock Task ---
class MockTask(Task):
    """Mock task that can throw errors."""

    id = "test_task_error"
    validate_per_step: bool = True  # Validate on every step to catch errors

    def __init__(self, should_error_on_step: int | None = None):
        self.should_error_on_step = should_error_on_step
        self.step_count = 0

    def setup(self, tool) -> tuple[Observation, dict]:
        self.step_count = 0
        return Observation.from_text("Test task: Complete 3 steps"), {}

    def validate_task(self, obs: Observation) -> tuple[float, dict]:
        self.step_count += 1
        # Error on specific step if configured
        if self.should_error_on_step and self.step_count == self.should_error_on_step:
            raise ValueError(f"Intentional error at step {self.should_error_on_step}")
        # Complete after 3 steps
        if self.step_count >= 3:
            return 1.0, {"done": True}
        return 0.0, {"done": False}

    def finished(self) -> bool:
        """Check if task is finished."""
        return self.step_count >= 3

    def filter_actions(self, actions: list[ActionSchema]) -> list[ActionSchema]:
        return actions


# --- Mock Agent ---
class ErrorAgentConfig(AgentConfig):
    """Agent config that creates an agent that throws errors."""

    name: str = "error_agent"
    error_on_step: int | None = None  # Step number to throw error

    def make(self, action_set: list[ActionSchema]) -> "ErrorAgent":
        return ErrorAgent(config=self, action_set=action_set)


class ErrorAgent(Agent):
    """Mock agent that can throw errors."""

    name = "ErrorAgent"

    def __init__(self, config: ErrorAgentConfig, action_set: list[ActionSchema]):
        super().__init__(config)
        self.config = config
        self.step_count = 0

    def step(self, obs: Observation) -> AgentOutput:
        self.step_count += 1
        # Throw error on specific step if configured
        if self.config.error_on_step and self.step_count == self.config.error_on_step:
            raise RuntimeError(f"Agent error at step {self.config.error_on_step}")
        return AgentOutput(actions=[Action(name="noop", arguments={})])


# --- Mock Benchmark ---
class MockBenchmark(Benchmark):
    """Mock benchmark for testing."""

    def __init__(self, tasks: list[Task], tool_config: ToolConfig):
        super().__init__(tool_config=tool_config)
        self._tasks = tasks

    def setup(self):
        pass

    def close(self):
        pass

    def load_tasks(self) -> list[Task]:
        return self._tasks


def test_agent_error():
    """Test that agent errors are stored in trajectory."""
    logger.info("=" * 60)
    logger.info("TEST 1: Agent Error Handling")
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "test_output"
        output_dir.mkdir()

        # Create agent that errors on step 1 (first agent step after initial env setup)
        agent_config = ErrorAgentConfig(error_on_step=1)
        tool_config = MockToolConfig()
        task = MockTask()

        env_config = EnvConfig(task=task, tool_config=tool_config)
        episode = Episode(
            id=0,
            output_dir=output_dir,
            agent_config=agent_config,
            env_config=env_config,
            exp_name="test_agent_error",
        )
        episode.storage.save_episode_config(episode.config)  # Save config for resumption test

        # Run episode - should catch agent error
        try:
            episode.run()
        except RuntimeError as e:
            logger.info(f"✓ Episode correctly raised error: {e}")

        # Check trajectory file (trajectory ID format: {task_id}_ep{id})
        traj_file = output_dir / "trajectories" / "test_task_error_ep0.jsonl"
        if traj_file.exists():
            logger.info(f"✓ Trajectory file created: {traj_file}")
            with open(traj_file) as f:
                lines = f.readlines()
                logger.info(f"✓ Trajectory has {len(lines)} steps")
                # Check for error in any step (should be in AgentOutput)
                import json

                found_error = False
                for line in lines:
                    step = json.loads(line)
                    output = step.get("output", {})
                    if output.get("_type") == "agentlab2.core.AgentOutput" and output.get("error"):
                        error = output["error"]
                        logger.info("✓ Error stored in trajectory:")
                        logger.info(f"  - Error type: {error['error_type']}")
                        logger.info(f"  - Error message: {error['exception_str']}")
                        logger.info(f"  - Has stack trace: {len(error['stack_trace']) > 0}")
                        found_error = True
                        break
                if not found_error:
                    logger.error("✗ Error not found in trajectory")
        else:
            logger.error(f"✗ Trajectory file not found: {traj_file}")

        # Check episode config was saved
        config_file = output_dir / "episode_configs" / "episode_0_task_test_task_error.json"
        if config_file.exists():
            logger.info(f"✓ Episode config saved: {config_file}")
        else:
            logger.error(f"✗ Episode config not found: {config_file}")


def test_env_error():
    """Test that environment errors are stored in trajectory."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Environment Error Handling")
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "test_output"
        output_dir.mkdir()

        # Create task that errors on step 2 (first env validation after agent action)
        agent_config = ErrorAgentConfig()  # No agent errors
        tool_config = MockToolConfig()
        task = MockTask(should_error_on_step=2)  # Error in env.step()

        env_config = EnvConfig(task=task, tool_config=tool_config)
        episode = Episode(
            id=1,
            output_dir=output_dir,
            agent_config=agent_config,
            env_config=env_config,
            exp_name="test_env_error",
        )

        # Run episode - should catch env error
        try:
            episode.run()
        except ValueError as e:
            logger.info(f"✓ Episode correctly raised error: {e}")

        # Check trajectory file (trajectory ID format: {task_id}_ep{id})
        traj_file = output_dir / "trajectories" / "test_task_error_ep1.jsonl"
        if traj_file.exists():
            logger.info(f"✓ Trajectory file created: {traj_file}")
            with open(traj_file) as f:
                lines = f.readlines()
                logger.info(f"✓ Trajectory has {len(lines)} steps")
                # Check for error in any step (should be in EnvironmentOutput)
                import json

                found_error = False
                for line in lines:
                    step = json.loads(line)
                    output = step.get("output", {})
                    if output.get("_type") == "agentlab2.core.EnvironmentOutput" and output.get("error"):
                        error = output["error"]
                        logger.info("✓ Error stored in trajectory:")
                        logger.info(f"  - Error type: {error['error_type']}")
                        logger.info(f"  - Error message: {error['exception_str']}")
                        found_error = True
                        break
                if not found_error:
                    logger.error("✗ Error not found in trajectory")
        else:
            logger.error(f"✗ Trajectory file not found: {traj_file}")


def test_experiment_resumption():
    """Test that failed episodes can be relaunched."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Experiment Resumption")
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "test_output"
        output_dir.mkdir()

        # Create experiment with one task that will fail
        agent_config = ErrorAgentConfig(error_on_step=1)  # Fails on first agent step
        tool_config = MockToolConfig()
        task = MockTask()
        benchmark = MockBenchmark([task], tool_config)

        exp = Experiment(
            name="test_resumption",
            output_dir=output_dir,
            agent_config=agent_config,
            benchmark=benchmark,
        )

        # Create episodes (this saves configs)
        episodes = exp.get_episodes_to_run()
        logger.info(f"✓ Created {len(episodes)} episodes")

        # Run first episode - it will fail
        try:
            episodes[0].run()
        except RuntimeError:
            logger.info("✓ Episode failed as expected")

        # Check that config was saved
        config_dir = output_dir / "episode_configs"
        config_files = list(config_dir.glob("*.json"))
        logger.info(f"✓ Found {len(config_files)} episode configs")

        # Test finding failed episodes via retry_failed flag
        exp.retry_failed = True
        failed_episodes = exp.get_episodes_to_run()
        exp.retry_failed = False
        if failed_episodes:
            logger.info(f"✓ Found {len(failed_episodes)} failed episodes to relaunch")
        else:
            logger.warning("⚠ No failed episodes found to relaunch")


if __name__ == "__main__":
    test_agent_error()
    test_env_error()
    test_experiment_resumption()
    logger.info("\n" + "=" * 60)
    logger.info("All tests completed!")
    logger.info("=" * 60)
