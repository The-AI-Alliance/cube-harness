import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel

from agentlab2.core import AgentOutput, Trajectory, TrajectoryStep
from agentlab2.episode_logs import get_log_path as get_episode_log_path

if TYPE_CHECKING:
    from agentlab2.episode import EpisodeConfig

logger = logging.getLogger(__name__)


class LLMCallRef(BaseModel):
    """Reference to an LLM call stored in a separate file."""

    llm_call_id: str


class Storage(Protocol):
    """Protocol for trajectory storage backends."""

    def save_trajectory(self, trajectory: Trajectory, allow_overwrite: bool = False) -> None:
        """Initialize storage for a trajectory and save metadata."""
        ...

    def save_step(self, step: TrajectoryStep, trajectory_id: str, step_num: int) -> None:
        """Append a single step to the trajectory."""
        ...

    def save_episode_config(self, episode_config: "EpisodeConfig") -> None:
        """Save episode configuration to disk for later resumption."""
        ...


def _trajectory_dir(output_dir: Path) -> Path:
    """Directory for trajectory files: use flat output_dir (no trajectories/ subdir)."""
    return output_dir


def _trajectory_path_legacy(output_dir: Path, trajectory_id: str, ext: str) -> Path | None:
    """Legacy path (trajectories/ subdir) for backward compatibility."""
    p = output_dir / "trajectories" / f"{trajectory_id}{ext}"
    return p if p.exists() else None


class FileStorage:
    """File-based storage for trajectories.

    All experiment files (trajectories, episode configs, llm_calls) are stored
    in the same directory (output_dir) so that one directory = one experiment.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self._current_traj_paths: dict[str, Path] = {}

    def save_trajectory(self, trajectory: Trajectory, allow_overwrite: bool = False) -> None:
        """Save the trajectory metadata and initialize the JSONL file.

        Args:
            trajectory: The trajectory to save.
            allow_overwrite: If True, archive existing trajectory files before saving.
                If False, raise FileExistsError when trajectory files already exist.
                Re-saves within the same session (e.g. updating end_time) are always allowed.
        """
        traj_dir = _trajectory_dir(self.output_dir)
        traj_dir.mkdir(parents=True, exist_ok=True)
        cur_path = traj_dir / trajectory.id

        # Check for pre-existing files from a previous run.
        # Skip the check if this trajectory was already created in this session (re-save for end_time update).
        is_resave = trajectory.id in self._current_traj_paths
        metadata_path = Path(f"{cur_path}.metadata.json")
        if not is_resave and metadata_path.exists():
            if not allow_overwrite:
                raise FileExistsError(
                    f"Trajectory '{trajectory.id}' already exists at {cur_path}. "
                    "Use allow_overwrite=True to archive the old trajectory and overwrite."
                )
            self._archive_trajectory(trajectory.id)

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

    def _archive_trajectory(self, trajectory_id: str) -> None:
        """Rename existing trajectory files with an archived timestamp suffix."""
        traj_dir = _trajectory_dir(self.output_dir)
        for ext in [".metadata.json", ".jsonl"]:
            old_path = traj_dir / f"{trajectory_id}{ext}"
            if old_path.exists():
                new_path = self.output_dir / f"{trajectory_id}.archived_{time.time()}{ext}"
                old_path.rename(new_path)
                logger.info(f"Archived {old_path.name} -> {new_path.name}")

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

        # Store llm_calls in same dir as other experiment files
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save each LLM call to a separate file
        llm_call_refs = []
        for llm_call in step.output.llm_calls:
            call_path = self.output_dir / f"{step_id}_{llm_call.id}.json"
            with open(call_path, "w") as f:
                f.write(llm_call.model_dump_json(indent=2))
            # Create a reference with just the id
            llm_call_refs.append(LLMCallRef(llm_call_id=llm_call.id))

        # Create a copy of the step with llm_calls replaced by references
        output_with_refs = step.output.model_copy(update={"llm_calls": llm_call_refs})
        return step.model_copy(update={"output": output_with_refs})

    def load_trajectory(self, trajectory_id: str) -> Trajectory:
        """Load a single trajectory by its ID."""
        traj_dir = _trajectory_dir(self.output_dir)
        metadata_path = traj_dir / f"{trajectory_id}.metadata.json"
        steps_path = traj_dir / f"{trajectory_id}.jsonl"
        # Backward compatibility: try legacy trajectories/ subdir
        if not metadata_path.exists():
            legacy = _trajectory_path_legacy(self.output_dir, trajectory_id, ".metadata.json")
            if legacy is not None:
                metadata_path = legacy
                steps_path = self.output_dir / "trajectories" / f"{trajectory_id}.jsonl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Trajectory metadata not found: {metadata_path}")

        with open(metadata_path) as f:
            trajectory_data = json.load(f)

        # TODO: remove legacy format support
        if "metadata" not in trajectory_data:
            trajectory_data = {"id": trajectory_id, "metadata": trajectory_data}

        if not steps_path.exists() and traj_dir == self.output_dir:
            steps_path = self.output_dir / "trajectories" / f"{trajectory_id}.jsonl"
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
        # Try flat dir first, then legacy llm_calls/ subdir
        resolved_calls = []
        for ref in llm_calls:
            # Check if this is a reference (only has 'id' key)
            if llm_call_id := ref.get("llm_call_id", None):
                call_path = self.output_dir / f"{step_id}_{llm_call_id}.json"
                if not call_path.exists():
                    call_path = self.output_dir / "llm_calls" / f"{step_id}_{llm_call_id}.json"
                if not call_path.exists():
                    raise FileNotFoundError(f"LLM call file not found: {call_path}")
                with open(call_path) as f:
                    resolved_calls.append(json.load(f))
            else:
                raise ValueError(f"Invalid LLM call reference format {ref}")

        step_data["output"]["llm_calls"] = resolved_calls
        return step_data

    def load_trajectory_metadata(self, trajectory_id: str) -> Trajectory:
        """Load only metadata (no steps) for fast experiment listing.

        Returns a Trajectory stub with steps=[] — significantly faster than
        load_trajectory() since it skips the JSONL file and all LLM call refs.
        """
        traj_dir = _trajectory_dir(self.output_dir)
        metadata_path = traj_dir / f"{trajectory_id}.metadata.json"
        if not metadata_path.exists():
            legacy = _trajectory_path_legacy(self.output_dir, trajectory_id, ".metadata.json")
            if legacy is not None:
                metadata_path = legacy
        if not metadata_path.exists():
            raise FileNotFoundError(f"Trajectory metadata not found: {metadata_path}")

        with open(metadata_path) as f:
            trajectory_data = json.load(f)

        # TODO: remove legacy format support
        if "metadata" not in trajectory_data:
            trajectory_data = {"id": trajectory_id, "metadata": trajectory_data}

        trajectory_data["steps"] = []
        return Trajectory.model_validate(trajectory_data)

    def load_all_trajectory_metadata(self) -> list[Trajectory]:
        """Load metadata stubs for all trajectories (no steps).

        Much faster than load_all_trajectories() — only reads *.metadata.json files.
        Each returned Trajectory has steps=[] until select_trajectory() loads it on demand.
        """
        traj_dir = _trajectory_dir(self.output_dir)
        seen_ids: set[str] = set()
        trajectories = []
        # Collect from flat dir then legacy trajectories/ subdir (dedupe by id)
        for search_dir in [traj_dir, self.output_dir / "trajectories"]:
            if not search_dir.exists():
                continue
            for metadata_file in search_dir.glob("*.metadata.json"):
                if ".archived_" in metadata_file.name:
                    continue
                trajectory_id = metadata_file.stem.replace(".metadata", "")
                if trajectory_id in seen_ids:
                    continue
                seen_ids.add(trajectory_id)
                try:
                    trajectories.append(self.load_trajectory_metadata(trajectory_id))
                except Exception as e:
                    logger.error(f"Failed to load trajectory metadata {trajectory_id}: {e}")

        return trajectories

    def list_trajectory_ids(self) -> list[str]:
        """List all non-archived trajectory IDs in the output directory."""
        seen: set[str] = set()
        result: list[str] = []
        for search_dir in [_trajectory_dir(self.output_dir), self.output_dir / "trajectories"]:
            if not search_dir.exists():
                continue
            for f in search_dir.glob("*.metadata.json"):
                if ".archived_" in f.name:
                    continue
                tid = f.stem.replace(".metadata", "")
                if tid not in seen:
                    seen.add(tid)
                    result.append(tid)
        return result

    def list_trajectory_ids_with_mtime(self) -> dict[str, float]:
        """List trajectory IDs mapped to their latest file modification time.

        Returns the max mtime across the .metadata.json and .jsonl files for each
        trajectory — cheap stat() calls only, no file reads. Used for change detection
        in live polling to avoid reloading trajectories that haven't changed.
        """
        result: dict[str, float] = {}
        for search_dir in [_trajectory_dir(self.output_dir), self.output_dir / "trajectories"]:
            if not search_dir.exists():
                continue
            for metadata_file in search_dir.glob("*.metadata.json"):
                if ".archived_" in metadata_file.name:
                    continue
                traj_id = metadata_file.stem.replace(".metadata", "")
                if traj_id in result:
                    continue
                mtime = metadata_file.stat().st_mtime
                jsonl_path = search_dir / f"{traj_id}.jsonl"
                if jsonl_path.exists():
                    mtime = max(mtime, jsonl_path.stat().st_mtime)
                result[traj_id] = mtime
        return result

    def get_log_path(self, trajectory_id: str) -> Path:
        return get_episode_log_path(self.output_dir, trajectory_id)

    def load_logs(self, trajectory_id: str) -> str:
        log_path = self.get_log_path(trajectory_id)
        if not log_path.exists():
            return ""
        return log_path.read_text()

    def has_logs(self, trajectory_id: str) -> bool:
        return self.get_log_path(trajectory_id).exists()

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

        trajectory_ids = self.list_trajectory_ids()
        trajectories = []
        for trajectory_id in trajectory_ids:
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
        self.output_dir.mkdir(parents=True, exist_ok=True)
        config_path = self.output_dir / f"episode_{episode_config.id}_task_{episode_config.task_id}.json"
        if config_path.exists():
            raise FileExistsError(
                f"Episode config already exists: {config_path}, are you trying to resume without setting the flag Experiment.resume?"
            )

        with open(config_path, "w") as f:
            f.write(episode_config.model_dump_json(indent=2, serialize_as_any=True))
        logger.info(f"Saved episode config to {config_path}")

    def load_episode_config(self, config_path: Path) -> "EpisodeConfig":
        """Load episode configuration from disk.

        Args:
            config_path: Path to the episode config JSON file.

        Returns:
            The loaded EpisodeConfig.
        """
        # Import here to avoid circular dependency at module level
        from agentlab2.episode import EpisodeConfig

        with open(config_path) as f:
            data = json.load(f)

        return EpisodeConfig.model_validate(data)

    def list_episode_configs(self) -> list[Path]:
        """List all episode config files in the output directory.

        Returns:
            List of paths to episode config files.
        """
        # Flat dir first, then legacy episode_configs/ subdir; dedupe by name so flat wins
        seen_names: set[str] = set()
        result: list[Path] = []
        for search_dir in [self.output_dir, self.output_dir / "episode_configs"]:
            if not search_dir.exists():
                continue
            for p in search_dir.glob("episode_*_task_*.json"):
                if p.name not in seen_names:
                    seen_names.add(p.name)
                    result.append(p)
        return result
