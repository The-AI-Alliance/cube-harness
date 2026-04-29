"""Atlas EvalLog: two-level structured evaluation records for community-scale agent benchmarking.

Two files per experiment:
    experiment_record.json  — ExperimentRecord (once per experiment): agent, benchmark, git provenance
    eval_log.jsonl          — EpisodeRecord per line: outcome, usage, trajectory summary

Records are plain JSON, no cube-harness dependency to read.

Classes:
    UsageSummary      — aggregated LLM token/cost stats across an episode
    AgentInfo         — agent descriptor: config, dependency versions, git provenance
    BenchmarkSubset   — benchmark subset descriptor for MNAR propensity correction
    JudgeConfig       — configuration of the judge LLM (optional)
    JudgeOutput       — per-episode judge assessment (optional)
    Verifier          — task verifier reference (optional)
    ExperimentRecord  — experiment-level record written to experiment_record.json
    EpisodeRecord     — episode-level record appended to eval_log.jsonl
    EvalLog           — two-level container with save/load
"""

import hashlib
import importlib.metadata
import json
import logging
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from cube.core import TypedBaseModel
from pydantic import Field

from cube_harness.core import Trajectory

logger = logging.getLogger(__name__)

_TRACKED_PACKAGES: list[str] = [
    "cube-harness",
    "cube",
    "litellm",
    "anthropic",
    "openai",
    "browsergym-core",
    "playwright",
    "pydantic",
    "ray",
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _collect_dependency_versions() -> dict[str, str]:
    """Return installed versions for all tracked packages that are present."""
    return {pkg: v for pkg in _TRACKED_PACKAGES if (v := _get_package_version(pkg)) is not None}


def _to_github_url(remote_url: str, commit: str) -> str | None:
    """Convert a git remote URL (HTTPS or SSH) to a permanent GitHub commit URL."""
    ssh = re.match(r"git@github\.com:(.+?)(?:\.git)?$", remote_url)
    if ssh:
        return f"https://github.com/{ssh.group(1)}/tree/{commit}"
    https = re.match(r"https://github\.com/(.+?)(?:\.git)?$", remote_url)
    if https:
        return f"https://github.com/{https.group(1)}/tree/{commit}"
    return None


def _get_git_info(cwd: str | None = None) -> tuple[str | None, str | None, bool | None]:
    """Return (commit_sha, github_permalink, is_dirty) for the repo at cwd.

    All three values are None when git is unavailable or cwd is not inside a repo.
    is_dirty is True when uncommitted changes exist (result may not reproduce exactly
    from git_commit alone).
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None, None, None

    try:
        remote = subprocess.check_output(
            ["git", "remote", "get-url", "origin"], cwd=cwd, stderr=subprocess.DEVNULL
        ).decode().strip()
        github_url = _to_github_url(remote, commit)
    except Exception:
        github_url = None

    try:
        is_dirty = subprocess.call(
            ["git", "diff", "--quiet", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
        ) != 0
    except Exception:
        is_dirty = None

    return commit, github_url, is_dirty


def _extract_llm_model(config_dict: dict) -> str | None:
    """Walk a serialized agent config dict looking for a model name field."""
    for key in ("model_name", "model", "llm_model"):
        if key in config_dict and isinstance(config_dict[key], str):
            return config_dict[key]
    for nested_key in ("llm_config", "llm"):
        nested = config_dict.get(nested_key)
        if isinstance(nested, dict):
            for key in ("model_name", "model"):
                if key in nested and isinstance(nested[key], str):
                    return nested[key]
    return None


def _extract_tool_names(tools: list[dict]) -> list[str]:
    """Extract action names from serialized action schemas.

    Handles both litellm format ({"type": "function", "function": {"name": ...}})
    and flat format ({"name": ...}).
    """
    names = []
    for tool in tools:
        fn = tool.get("function", {})
        if isinstance(fn, dict) and "name" in fn:
            names.append(fn["name"])
        elif "name" in tool:
            names.append(tool["name"])
    return names


def _extract_error_type(trajectory: Trajectory) -> str | None:
    """Return the error_type of the first StepError in the trajectory, or None."""
    for step in trajectory.steps:
        if hasattr(step.output, "error") and step.output.error is not None:
            return step.output.error.error_type
    return None


# ---------------------------------------------------------------------------
# Public models
# ---------------------------------------------------------------------------


class UsageSummary(TypedBaseModel):
    """Aggregated LLM token usage and cost across a complete episode."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    cache_creation_tokens: int = 0
    total_cost_usd: float = 0.0
    n_llm_calls: int = 0

    @classmethod
    def from_summary_stats(cls, stats: dict | None) -> "UsageSummary":
        """Build from the summary_stats dict stored on a Trajectory."""
        if not stats:
            return cls()
        prompt = stats.get("prompt_tokens", 0)
        completion = stats.get("completion_tokens", 0)
        return cls(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
            cached_tokens=stats.get("cached_tokens", 0),
            cache_creation_tokens=stats.get("cache_creation_tokens", 0),
            total_cost_usd=stats.get("cost", 0.0),
            n_llm_calls=stats.get("total_llm_calls", 0),
        )


class AgentInfo(TypedBaseModel):
    """Agent descriptor for Atlas embedding and reproducibility.

    Tools are NOT included here — they vary per episode due to task-level action filtering.
    See EpisodeRecord.tool_names for the per-episode tool list.
    """

    agent_id: str = Field(description="SHA-256 of the serialized agent config — stable unique identifier across runs.")
    config_type: str = Field(description="Agent config class name (from _type discriminator field).")
    config: dict = Field(description="Full serialized agent config (model_dump with serialize_as_any=True).")
    llm_model: str | None = Field(default=None, description="LLM model name extracted from config.")
    framework_version: str = Field(description="cube-harness version at eval time.")
    dependency_versions: dict[str, str] = Field(
        default_factory=dict,
        description="Installed versions of tracked packages (cube-harness, litellm, anthropic, openai, ...).",
    )
    git_commit: str | None = Field(default=None, description="Git SHA-1 of the repo HEAD at eval time.")
    git_remote_url: str | None = Field(
        default=None,
        description="Permanent GitHub URL pointing to the exact commit (tree view). None if not on GitHub.",
    )
    git_is_dirty: bool | None = Field(
        default=None,
        description=(
            "True when uncommitted changes exist at eval time — result may not reproduce exactly "
            "from git_commit alone. None when git info is unavailable."
        ),
    )
    description: str | None = Field(
        default=None,
        description="Free-form prose description of the agent for Atlas LLM embedding warm-start.",
    )

    @classmethod
    def from_agent_config(
        cls,
        agent_config: Any,
        git_cwd: str | None = None,
    ) -> "AgentInfo":
        """Build AgentInfo from an agent config object.

        Args:
            agent_config: Any AgentConfig (TypedBaseModel subclass).
            git_cwd: Working directory for git commands. Defaults to CWD.
        """
        harness_version = _get_package_version("cube-harness") or "unknown"
        config_dict = json.loads(agent_config.model_dump_json(serialize_as_any=True))
        agent_id = hashlib.sha256(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()
        config_type = config_dict.get("_type", type(agent_config).__name__)
        llm_model = _extract_llm_model(config_dict)
        git_commit, git_remote_url, git_is_dirty = _get_git_info(cwd=git_cwd)

        return cls(
            agent_id=agent_id,
            config_type=config_type,
            config=config_dict,
            llm_model=llm_model,
            framework_version=harness_version,
            dependency_versions=_collect_dependency_versions(),
            git_commit=git_commit,
            git_remote_url=git_remote_url,
            git_is_dirty=git_is_dirty,
        )


class BenchmarkSubset(TypedBaseModel):
    """Benchmark subset descriptor for MNAR propensity correction.

    Automatically derived from the benchmark object. The name field captures any subset
    suffix applied via subset_from_glob (e.g., "[level=l1]") or subset_from_list.
    n_tasks is the denominator for computing completion rate without requiring the benchmark.
    """

    name: str = Field(description="Benchmark name including any subset suffix (benchmark_metadata.name).")
    n_tasks: int = Field(description="Total tasks in this subset — denominator for completion rate.")
    filter: str | None = Field(
        default=None,
        description="Glob expression if the subset was created via subset_from_glob.",
    )

    @classmethod
    def from_benchmark(cls, benchmark: Any) -> "BenchmarkSubset":
        """Derive BenchmarkSubset from a cube Benchmark object."""
        bm_metadata = getattr(benchmark, "benchmark_metadata", None)
        name = bm_metadata.name if bm_metadata else "unknown"
        n_tasks = len(getattr(benchmark, "task_metadata", {}))
        return cls(name=name, n_tasks=n_tasks)


class JudgeConfig(TypedBaseModel):
    """Configuration of the LLM judge used for post-hoc episode assessment."""

    model: str = Field(description="Judge model identifier (e.g. 'claude-opus-4-7').")
    prompt_version: str = Field(description="Version or hash of the judge prompt template.")
    judged_at: str | None = Field(default=None, description="ISO-8601 timestamp when judging was run.")


class JudgeOutput(TypedBaseModel):
    """Per-episode assessment from a post-hoc LLM judge."""

    difficulty: str | None = Field(default=None, description="Estimated task difficulty (free-form or enum).")
    feasible: bool | None = Field(default=None, description="Whether the task was deemed completable by the judge.")
    failure_root_cause: str | None = Field(default=None, description="Short description of why the agent failed.")


class Verifier(TypedBaseModel):
    """Task verifier reference for reproducibility and post-hoc inspection."""

    ref: str | None = Field(
        default=None,
        description="Permanent GitHub URL pointing to the verifier function at the exact commit.",
    )
    source: str | None = Field(
        default=None,
        description="Verifier source code at eval time (for auditing without git access).",
    )


class ExperimentRecord(TypedBaseModel):
    """Experiment-level record. Written once to experiment_record.json.

    Contains all fields shared across every episode: agent description, benchmark
    metadata, git provenance. EpisodeRecord links to this via experiment_id.
    """

    experiment_id: str = Field(description="SHA-256(experiment_name + output_dir)[:16] — stable within a run.")
    experiment_name: str = Field(description="Experiment name as set in Experiment.name.")
    timestamp: float = Field(description="Experiment export time as Unix timestamp.")
    framework_version: str = Field(description="cube-harness version.")
    agent: AgentInfo = Field(description="Agent descriptor (config, dependency versions, git provenance).")
    benchmark_name: str = Field(description="Benchmark name from benchmark_metadata.name.")
    benchmark_version: str | None = Field(default=None, description="Benchmark version string.")
    benchmark_subset: BenchmarkSubset = Field(description="Subset descriptor for MNAR propensity correction.")
    judge_config: JudgeConfig | None = Field(
        default=None,
        description="Judge configuration if a post-hoc LLM judge was run on these episodes.",
    )

    @classmethod
    def from_experiment(
        cls,
        exp_name: str,
        output_dir: Path,
        agent_config: Any,
        benchmark: Any,
        git_cwd: str | None = None,
    ) -> "ExperimentRecord":
        """Build ExperimentRecord from experiment parameters."""
        experiment_id = hashlib.sha256(f"{exp_name}{output_dir}".encode()).hexdigest()[:16]
        harness_version = _get_package_version("cube-harness") or "unknown"
        agent_info = AgentInfo.from_agent_config(agent_config, git_cwd=git_cwd)
        bm_metadata = getattr(benchmark, "benchmark_metadata", None)
        bm_name = bm_metadata.name if bm_metadata else "unknown"
        bm_version = getattr(bm_metadata, "version", None) if bm_metadata else None

        return cls(
            experiment_id=experiment_id,
            experiment_name=exp_name,
            timestamp=time.time(),
            framework_version=harness_version,
            agent=agent_info,
            benchmark_name=bm_name,
            benchmark_version=bm_version,
            benchmark_subset=BenchmarkSubset.from_benchmark(benchmark),
        )


class EpisodeRecord(TypedBaseModel):
    """Episode-level record. One line per episode in eval_log.jsonl.

    Links to ExperimentRecord via experiment_id. Contains all episode-specific fields:
    task identity, per-episode tool list, outcome, usage, and optional judge output.
    """

    experiment_id: str = Field(description="FK to ExperimentRecord.experiment_id.")
    task_id: str = Field(description="Unique task identifier within the benchmark.")
    task_version_hash: str | None = Field(
        default=None,
        description="SHA-256 of TaskConfig JSON. Detects task drift across benchmark versions.",
    )
    seed: int | None = Field(default=None, description="Random seed for this task instance.")
    split: str | None = Field(default=None, description="Dataset split: 'train', 'val', or 'test'.")
    task_description: str | None = Field(
        default=None,
        description="Abstract task description from TaskMetadata.abstract_description.",
    )
    tool_names: list[str] = Field(
        default_factory=list,
        description=(
            "Action names available during this episode. Episode-specific: same agent gets "
            "different tools on different tasks due to task-level action filtering."
        ),
    )
    success: bool = Field(description="True when final reward > 0.")
    reward: float = Field(description="Final reward from the last EnvironmentOutput.")
    error_type: str | None = Field(
        default=None,
        description="Exception class name if any step raised an error. None for clean episodes.",
    )
    n_steps: int = Field(description="Total trajectory steps (agent + env combined).")
    n_agent_steps: int = Field(description="Agent steps (LLM decision turns).")
    n_env_steps: int = Field(description="Environment steps (tool executions).")
    wall_time_s: float | None = Field(default=None, description="Episode wall-clock duration in seconds.")
    usage: UsageSummary = Field(
        default_factory=UsageSummary,
        description="Aggregated LLM token usage and cost for the episode.",
    )
    trajectory_id: str = Field(description="Trajectory ID as stored on disk.")
    timestamp: float = Field(description="Episode start time as Unix timestamp.")
    verifier: Verifier | None = Field(
        default=None,
        description="Task verifier reference for reproducibility and post-hoc inspection.",
    )
    judge_output: JudgeOutput | None = Field(
        default=None,
        description="Per-episode LLM judge assessment (difficulty, feasibility, failure root cause).",
    )

    @classmethod
    def from_trajectory(
        cls,
        trajectory: Trajectory,
        experiment_id: str,
        task_metadata: Any | None = None,
        task_config: Any | None = None,
    ) -> "EpisodeRecord":
        """Assemble an EpisodeRecord from a completed trajectory."""
        task_id = trajectory.metadata.get("task_id", "")
        action_schemas: list[dict] = trajectory.metadata.get("action_schemas", [])
        tool_names = _extract_tool_names(action_schemas)

        last_env = trajectory.last_env_step()
        reward = last_env.reward
        stats = trajectory.summary_stats or {}

        wall_time_s: float | None = None
        if trajectory.start_time is not None and trajectory.end_time is not None:
            wall_time_s = trajectory.end_time - trajectory.start_time

        task_version_hash: str | None = None
        seed: int | None = None
        if task_config is not None:
            task_version_hash = hashlib.sha256(
                task_config.model_dump_json(serialize_as_any=True).encode()
            ).hexdigest()
            seed = getattr(task_config, "seed", None)

        split: str | None = None
        task_description: str | None = None
        if task_metadata is not None:
            split = getattr(task_metadata, "split", None)
            task_description = getattr(task_metadata, "abstract_description", None) or None

        return cls(
            experiment_id=experiment_id,
            task_id=task_id,
            task_version_hash=task_version_hash,
            seed=seed,
            split=split,
            task_description=task_description,
            tool_names=tool_names,
            success=reward > 0,
            reward=reward,
            error_type=_extract_error_type(trajectory),
            n_steps=len(trajectory.steps),
            n_agent_steps=stats.get("n_agent_steps", trajectory.n_agent_steps),
            n_env_steps=stats.get("n_env_steps", trajectory.n_env_steps),
            wall_time_s=wall_time_s,
            usage=UsageSummary.from_summary_stats(stats),
            trajectory_id=trajectory.id,
            timestamp=trajectory.start_time or 0.0,
        )


class EvalLog(TypedBaseModel):
    """Two-level eval log container with JSONL serialization.

    Experiment-level data goes to experiment_record.json (written once).
    Episode-level data goes to eval_log.jsonl (one JSON object per line).

    Both files are plain JSON, readable by any framework without a cube-harness dependency.
    """

    experiment: ExperimentRecord
    episodes: list[EpisodeRecord] = Field(default_factory=list)

    def save(self, output_dir: Path) -> None:
        """Write experiment_record.json and eval_log.jsonl to output_dir."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        exp_path = output_dir / "experiment_record.json"
        exp_path.write_text(self.experiment.model_dump_json(indent=2))
        episodes_path = output_dir / "eval_log.jsonl"
        with open(episodes_path, "w") as f:
            for record in self.episodes:
                f.write(record.model_dump_json() + "\n")
        logger.info(f"Saved experiment record to {exp_path}")
        logger.info(f"Saved {len(self.episodes)} episode records to {episodes_path}")

    @classmethod
    def load(cls, output_dir: Path) -> "EvalLog":
        """Load experiment_record.json and eval_log.jsonl from output_dir."""
        output_dir = Path(output_dir)
        experiment = ExperimentRecord.model_validate_json((output_dir / "experiment_record.json").read_text())
        episodes: list[EpisodeRecord] = []
        with open(output_dir / "eval_log.jsonl") as f:
            for line in f:
                line = line.strip()
                if line:
                    episodes.append(EpisodeRecord.model_validate_json(line))
        return cls(experiment=experiment, episodes=episodes)

    @staticmethod
    def append_episode(record: EpisodeRecord, path: Path) -> None:
        """Append a single EpisodeRecord to a JSONL file (streaming mode).

        Suitable for writing records immediately after each episode without holding the full
        log in memory. Not safe for concurrent multi-process writes without external file locking.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(record.model_dump_json() + "\n")
