import json
import logging
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from agentlab2.core import AgentOutput, Trajectory, TrajectoryStep

logger = logging.getLogger(__name__)

# Forward reference to avoid circular import
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentlab2.episode import EpisodeConfig


class LLMCallRef(BaseModel):
    """Reference to an LLM call stored in a separate file."""

    llm_call_id: str


class Storage(Protocol):
    """Protocol for trajectory storage backends."""

    def save_trajectory(self, trajectory: Trajectory) -> None:
        """Initialize storage for a trajectory and save metadata."""
        ...

    def save_step(self, step: TrajectoryStep, trajectory_id: str, step_num: int) -> None:
        """Append a single step to the trajectory."""
        ...


class FileStorage:
    """File-based storage for trajectories."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self._current_traj_paths: dict[str, Path] = {}

    def save_trajectory(self, trajectory: Trajectory) -> None:
        """Save the trajectory metadata and initialize the JSONL file."""
        traj_dir = self.output_dir / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)
        cur_path = traj_dir / trajectory.id
        self._current_traj_paths[trajectory.id] = cur_path
        with open(f"{cur_path}.metadata.json", "w") as f:
            # Serialize entire trajectory excluding steps
            trajectory_data = trajectory.model_dump(exclude={"steps"})
            f.write(json.dumps(trajectory_data, indent=2))

        # Create empty file for appending steps later
        with open(f"{cur_path}.jsonl", "w") as f:
            pass

        # Save initial steps
        for i, step in enumerate(trajectory.steps):
            self._append_step(step, trajectory.id, i)

        logger.info(f"Saved trajectory to {cur_path}")

    def save_step(self, step: TrajectoryStep, trajectory_id: str, step_num: int) -> None:
        """Append a single step to the trajectory JSONL file."""
        if trajectory_id not in self._current_traj_paths:
            raise ValueError("Trajectory path not set. Call save_trajectory first.")
        try:
            self._append_step(step, trajectory_id, step_num)
        except Exception as e:
            logger.exception(f"Error saving step to trajectory {self._current_traj_paths[trajectory_id]}: {e}")
            raise e

    def _append_step(self, step: TrajectoryStep, trajectory_id: str, step_num: int) -> None:
        """Internal method to append a step to the JSONL file."""
        step_to_save = step
        cur_path = self._current_traj_paths[trajectory_id]
        if isinstance(step.output, AgentOutput) and step.output.llm_calls:
            step_to_save = self._extract_llm_calls(step, f"{trajectory_id}_step{step_num:03d}")

        with open(f"{cur_path}.jsonl", "a") as f:
            line = step_to_save.model_dump_json(serialize_as_any=True)
            f.write(f"{line}\n")

    def _extract_llm_calls(self, step: TrajectoryStep, step_id: str) -> TrajectoryStep:
        """Extract LLM calls to separate files and return step with references only."""
        assert isinstance(step.output, AgentOutput)

        llm_calls_dir = self.output_dir / "llm_calls"
        llm_calls_dir.mkdir(parents=True, exist_ok=True)

        # Save each LLM call to a separate file
        llm_call_refs = []
        for llm_call in step.output.llm_calls:
            call_path = llm_calls_dir / f"{step_id}_{llm_call.id}.json"
            with open(call_path, "w") as f:
                f.write(llm_call.model_dump_json(indent=2))
            # Create a reference with just the id
            llm_call_refs.append(LLMCallRef(llm_call_id=llm_call.id))

        # Create a copy of the step with llm_calls replaced by references
        output_with_refs = step.output.model_copy(update={"llm_calls": llm_call_refs})
        return step.model_copy(update={"output": output_with_refs})

    def load_trajectory(self, trajectory_id: str) -> Trajectory:
        """Load a single trajectory by its ID."""
        traj_dir = self.output_dir / "trajectories"
        metadata_path = traj_dir / f"{trajectory_id}.metadata.json"
        steps_path = traj_dir / f"{trajectory_id}.jsonl"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Trajectory metadata not found: {metadata_path}")

        with open(metadata_path) as f:
            trajectory_data = json.load(f)

        # TODO: remove legacy format support
        if "metadata" not in trajectory_data:
            trajectory_data = {"id": trajectory_id, "metadata": trajectory_data}

        steps: list[TrajectoryStep] = []
        if steps_path.exists():
            with open(steps_path) as f:
                for i, line in enumerate(f):
                    if line.strip():
                        step_data = json.loads(line)
                        step_data = self._resolve_llm_call_refs(step_data, trajectory_id, i)
                        if "output" not in step_data:
                            if "obs" in step_data:
                                # Legacy format where step is just EnvironmentOutput
                                step_data = {"output": step_data}
                            elif "actions" in step_data:
                                # Legacy format where step is just AgentOutput
                                step_data = {"output": step_data}
                        step = TrajectoryStep.model_validate(step_data)
                        steps.append(step)

        trajectory_data["steps"] = steps
        return Trajectory.model_validate(trajectory_data)

    def _resolve_llm_call_refs(self, step_data: dict, trajectory_id: str, step_num: int) -> dict:
        """Resolve LLM call references by loading full LLMCall data from files."""
        output = step_data.get("output", {})
        llm_calls = output.get("llm_calls", [])

        if not llm_calls:
            return step_data

        step_id = f"{trajectory_id}_step{step_num:03d}"
        llm_calls_dir = self.output_dir / "llm_calls"

        resolved_calls = []
        for ref in llm_calls:
            # Check if this is a reference (only has 'id' key)
            if llm_call_id := ref.get("llm_call_id", None):
                call_path = llm_calls_dir / f"{step_id}_{llm_call_id}.json"
                if not call_path.exists():
                    raise FileNotFoundError(f"LLM call file not found: {call_path}")
                with open(call_path) as f:
                    resolved_calls.append(json.load(f))
            else:
                raise ValueError(f"Invalid LLM call reference format {ref}")

        step_data["output"]["llm_calls"] = resolved_calls
        return step_data

    def load_all_trajectories(self, exp_dir: str | Path | None = None) -> list[Trajectory]:
        """Load all trajectories from an experiment directory.

        Args:
            exp_dir: The experiment directory to load from. If None, uses self.output_dir.

        Returns:
            List of all trajectories found in the directory.
        """
        if exp_dir is not None:
            storage = FileStorage(exp_dir)
            return storage.load_all_trajectories()

        traj_dir = self.output_dir / "trajectories"
        if not traj_dir.exists():
            return []

        trajectories = []
        # Find all metadata files and extract trajectory IDs
        for metadata_file in traj_dir.glob("*.metadata.json"):
            trajectory_id = metadata_file.stem.replace(".metadata", "")
            try:
                trajectory = self.load_trajectory(trajectory_id)
                trajectories.append(trajectory)
            except Exception as e:
                logger.error(f"Failed to load trajectory {trajectory_id}: {e}")

        return trajectories

    def save_episode_config(self, episode_config: "EpisodeConfig") -> None:
        """Save episode configuration to disk for later resumption.

        Args:
            episode_config: The episode configuration to save.
        """
        config_dir = self.output_dir / "episode_configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / f"episode_{episode_config.id}_task_{episode_config.task_id}.json"

        with open(config_path, "w") as f:
            f.write(episode_config.model_dump_json(indent=2))
        logger.info(f"Saved episode config to {config_path}")

    def load_episode_config(self, config_path: Path) -> "EpisodeConfig":
        """Load episode configuration from disk.

        Args:
            config_path: Path to the episode config JSON file.

        Returns:
            The loaded EpisodeConfig.
        """
        from agentlab2.episode import EpisodeConfig

        with open(config_path) as f:
            data = json.load(f)

        return EpisodeConfig.model_validate(data)

    def list_episode_configs(self) -> list[Path]:
        """List all episode config files in the output directory.

        Returns:
            List of paths to episode config files.
        """
        config_dir = self.output_dir / "episode_configs"
        if not config_dir.exists():
            return []
        return list(config_dir.glob("episode_*_task_*.json"))
