import json
from pathlib import Path
from typing import Iterator

from cube_harness.core import Trajectory, TrajectoryStep
from cube_harness.storage import FileStorage


class EpisodeResult:

    def __init__(self, episode_dir: Path, storage: FileStorage) -> None:
        self._dir = episode_dir
        self._storage = storage
        self._metadata: Trajectory | None = None
        self._steps: dict[int, TrajectoryStep] = {}
        self._traj_id: str | None = None

    @property
    def trajectory_id(self) -> str:
        if self._traj_id is None:
            self._traj_id = self.metadata.id
        return self._traj_id

    @property
    def metadata(self) -> Trajectory:
        if self._metadata is None:
            metadata_path = self._dir / "episode.metadata.json"
            with open(metadata_path) as f:
                data = json.load(f)
            data["steps"] = []
            self._metadata = Trajectory.model_validate(data)
        return self._metadata

    @property
    def summary_stats(self) -> dict | None:
        return self.metadata.summary_stats

    def __len__(self) -> int:
        steps_dir = self._dir / "steps"
        if not steps_dir.exists():
            return 0
        return sum(1 for _ in steps_dir.iterdir())

    def __getitem__(self, index: int) -> TrajectoryStep:
        if index not in self._steps:
            self._steps[index] = self._storage.load_step(self.trajectory_id, index)
        return self._steps[index]

    def __iter__(self) -> Iterator[TrajectoryStep]:
        for i in range(len(self)):
            yield self[i]

    def load_full(self) -> Trajectory:
        return self._storage.load_trajectory(self.trajectory_id)


class ExperimentResult:

    def __init__(self, exp_dir: str | Path) -> None:
        self._dir = Path(exp_dir)
        self._storage = FileStorage(self._dir)
        self._episodes: dict[str, EpisodeResult] | None = None

    @property
    def episodes(self) -> dict[str, EpisodeResult]:
        if self._episodes is None:
            self._episodes = {}
            episodes_dir = self._dir / "episodes"
            if episodes_dir.exists():
                for ep_dir in sorted(episodes_dir.iterdir()):
                    if ep_dir.is_dir() and ".archived_" not in ep_dir.name:
                        metadata_path = ep_dir / "episode.metadata.json"
                        if metadata_path.exists():
                            with open(metadata_path) as f:
                                data = json.load(f)
                            traj_id = data.get("id", ep_dir.name)
                            self._episodes[traj_id] = EpisodeResult(ep_dir, self._storage)
        return self._episodes

    @property
    def summary(self) -> dict | None:
        path = self._dir / "experiment_summary.json"
        if path.exists():
            return json.loads(path.read_text())
        return None

    def to_df(self):
        from cube_harness.analyze.inspect_results import trajectories_to_df

        trajs = self._storage.load_all_trajectory_metadata()
        return trajectories_to_df(trajs)
