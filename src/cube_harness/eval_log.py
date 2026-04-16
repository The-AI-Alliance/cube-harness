"""Atlas EvalLog: structured evaluation records for community-scale agent benchmarking.

Implements the TaskEvalRecord schema required by Project ATLAS for populating the
Community EvalLog. Records are emitted after each episode and exported as JSONL,
making them compatible with any framework that can parse JSON.

The Pydantic models are for internal convenience; the on-disk format is plain JSON,
so downstream frameworks (non-Python, non-cube) can consume it without any dependency
on cube-harness.

Classes:
    UsageSummary   — aggregated LLM token usage and cost across an episode
    AgentInfo      — full agent descriptor: config, capabilities, dependencies, git provenance
    TaskInfo       — full task descriptor: benchmark metadata, task metadata, content hash
    TaskEvalRecord — complete evaluation record ready for Atlas ingestion
    EvalLog        — collection of records with JSONL save/load and streaming append
"""

import hashlib
import importlib.metadata
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any

from cube.core import EnvironmentOutput, TypedBaseModel
from pydantic import Field

from cube_harness.core import Trajectory

logger = logging.getLogger(__name__)

# Key packages whose installed versions are captured for reproducibility.
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


def _extract_first_observation_text(trajectory: Trajectory) -> str | None:
    """Extract text content from the first EnvironmentOutput in the trajectory.

    This is the actual task instruction as the agent received it.
    """
    for step in trajectory.steps:
        if isinstance(step.output, EnvironmentOutput):
            texts = [
                content.data
                for content in step.output.obs.contents
                if isinstance(getattr(content, "data", None), str)
            ]
            return "\n".join(texts) if texts else None
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
    """Complete agent descriptor for Atlas embedding and reproducibility.

    Designed for maximum downstream utility: includes structured fields (config, tools,
    versions) that can be queried or used to synthesize a prose description for LLM
    embedding, without requiring access to the agent's source code.
    """

    # Identity
    agent_id: str = Field(
        description="SHA-256 of the serialized agent config — stable unique identifier across runs."
    )

    # Config (structured, queryable)
    config_type: str = Field(description="Agent config class name (from _type discriminator field).")
    config: dict = Field(description="Full serialized agent config (model_dump with serialize_as_any=True).")
    llm_model: str | None = Field(default=None, description="LLM model name extracted from config.")

    # Capabilities — populated per-episode from the action set given to this agent.
    # The same agent gets different tools on different tasks; these reflect THIS episode.
    tools: list[dict] = Field(
        default_factory=list,
        description="Full action schemas in litellm function-call format (type, function.name, description, parameters).",
    )
    tool_names: list[str] = Field(
        default_factory=list,
        description="Action names for quick lookup — derived from tools.",
    )

    # Runtime environment
    framework_version: str = Field(description="cube-harness version at eval time.")
    dependency_versions: dict[str, str] = Field(
        default_factory=dict,
        description="Installed versions of key packages (cube-harness, litellm, anthropic, openai, ...).",
    )

    # Code provenance — enables Atlas to link records to an exact, auditable code state.
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

    # Optional prose — intended for Atlas LLM embedding warm-start.
    # Can be human-authored or synthesized from the structured fields above.
    description: str | None = Field(
        default=None,
        description=(
            "Free-form prose description of the agent (e.g. 'ReAct agent using gpt-4o, "
            "browser tools, no memory, 16k context window'). "
            "Used by Atlas to embed the agent in latent space via a frontier LLM."
        ),
    )

    @classmethod
    def from_agent_config(
        cls,
        agent_config: Any,
        action_schemas: list[dict] | None = None,
        git_cwd: str | None = None,
    ) -> "AgentInfo":
        """Build AgentInfo from a serialized agent config.

        Args:
            agent_config: Any AgentConfig (TypedBaseModel subclass).
            action_schemas: Pre-serialized action schemas in litellm format, as stored
                            in trajectory.metadata["action_schemas"]. If provided, no
                            task instantiation is needed to capture capabilities.
            git_cwd: Working directory for git commands. Defaults to CWD.
        """
        harness_version = _get_package_version("cube-harness") or "unknown"
        config_dict = json.loads(agent_config.model_dump_json(serialize_as_any=True))
        agent_id = hashlib.sha256(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()
        config_type = config_dict.get("_type", type(agent_config).__name__)
        llm_model = _extract_llm_model(config_dict)
        tools = action_schemas or []
        tool_names = _extract_tool_names(tools)
        git_commit, git_remote_url, git_is_dirty = _get_git_info(cwd=git_cwd)

        return cls(
            agent_id=agent_id,
            config_type=config_type,
            config=config_dict,
            llm_model=llm_model,
            tools=tools,
            tool_names=tool_names,
            framework_version=harness_version,
            dependency_versions=_collect_dependency_versions(),
            git_commit=git_commit,
            git_remote_url=git_remote_url,
            git_is_dirty=git_is_dirty,
        )

    def with_action_schemas(self, action_schemas: list[dict]) -> "AgentInfo":
        """Return a copy of this AgentInfo with the given action schemas applied."""
        return self.model_copy(update={"tools": action_schemas, "tool_names": _extract_tool_names(action_schemas)})


class TaskInfo(TypedBaseModel):
    """Complete task descriptor for Atlas embedding and reproducibility.

    Covers both the benchmark-level context (name, version, authors) and the specific
    task instance (id, seed, split, description). The first_observation_text field
    carries the actual instruction the agent saw — crucial for Atlas task embedding.
    """

    # Benchmark identity
    benchmark_id: str = Field(description="Stable machine-readable benchmark identifier (lowercased name).")
    benchmark_name: str = Field(description="Human-readable benchmark name (from BenchmarkMetadata).")
    benchmark_version: str | None = Field(default=None, description="Benchmark version string.")
    benchmark_description: str | None = Field(default=None, description="Benchmark description.")
    benchmark_authors: list[str] = Field(default_factory=list, description="Benchmark author names.")
    benchmark_tags: list[str] = Field(
        default_factory=list, description="Benchmark tags (domain, modality, difficulty, etc.)."
    )

    # Task identity
    task_id: str = Field(description="Unique task identifier within the benchmark.")
    task_version_hash: str | None = Field(
        default=None,
        description=(
            "SHA-256 of the serialized TaskConfig JSON. Detects task drift: if a benchmark update "
            "silently changes a task, the hash changes and Atlas can flag the record as stale."
        ),
    )
    seed: int | None = Field(default=None, description="Random seed for this task instance.")
    split: str | None = Field(default=None, description="Dataset split: 'train', 'val', or 'test'.")

    # Task description (layered, as Atlas specifies)
    abstract_description: str | None = Field(
        default=None,
        description=(
            "Broad task description from TaskMetadata.abstract_description — for searching and "
            "filtering only. Not the specific goal shown to the agent."
        ),
    )
    first_observation_text: str | None = Field(
        default=None,
        description=(
            "Text content of the first observation as the agent received it — the actual task "
            "instruction. This is the primary field for Atlas task embedding."
        ),
    )
    recommended_max_steps: int | None = Field(
        default=None, description="Benchmark-recommended step budget for this task."
    )
    extra_info: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional task metadata from TaskMetadata.extra_info (difficulty, domain, etc.).",
    )

    # Privileged — not shown to the agent during evaluation.
    eval_function_ref: str | None = Field(
        default=None,
        description=(
            "Qualified reference to the evaluation function. "
            "Privileged: the agent never sees this; the EvalLog does. "
            "Format: 'module.TaskConfigClass.make() → Task.evaluate'."
        ),
    )

    @classmethod
    def from_trajectory_and_metadata(
        cls,
        trajectory: Trajectory,
        benchmark_metadata: Any | None = None,
        task_metadata: Any | None = None,
        task_config: Any | None = None,
    ) -> "TaskInfo":
        """Build TaskInfo from a completed trajectory and optional rich metadata.

        Args:
            trajectory: Completed episode trajectory (provides task_id, first obs text).
            benchmark_metadata: cube.benchmark.BenchmarkMetadata (provides name, version, ...).
            task_metadata: cube.task.TaskMetadata for this task (provides split, description, ...).
            task_config: cube.task.TaskConfig for this episode (provides seed, content hash).
        """
        task_id: str = trajectory.metadata.get("task_id", "")

        if benchmark_metadata is not None:
            bm_name: str = benchmark_metadata.name
            bm_id: str = bm_name.lower().replace(" ", "-").replace("+", "plus")
            bm_version: str | None = benchmark_metadata.version
            bm_description: str | None = benchmark_metadata.description
            bm_authors: list[str] = list(benchmark_metadata.authors)
            bm_tags: list[str] = list(benchmark_metadata.tags)
        else:
            bm_name = trajectory.metadata.get("benchmark_name", "unknown")
            bm_id = bm_name.lower().replace(" ", "-")
            bm_version = None
            bm_description = None
            bm_authors = []
            bm_tags = []

        split: str | None = None
        abstract_description: str | None = None
        recommended_max_steps: int | None = None
        extra_info: dict[str, Any] = {}
        if task_metadata is not None:
            split = task_metadata.split
            abstract_description = task_metadata.abstract_description or None
            recommended_max_steps = task_metadata.recommended_max_steps
            extra_info = dict(task_metadata.extra_info)

        task_version_hash: str | None = None
        seed: int | None = None
        eval_function_ref: str | None = None
        if task_config is not None:
            task_version_hash = hashlib.sha256(
                task_config.model_dump_json(serialize_as_any=True).encode()
            ).hexdigest()
            seed = task_config.seed
            tc_module = type(task_config).__module__
            tc_name = type(task_config).__name__
            eval_function_ref = f"{tc_module}.{tc_name}.make() → Task.evaluate"

        return cls(
            benchmark_id=bm_id,
            benchmark_name=bm_name,
            benchmark_version=bm_version,
            benchmark_description=bm_description,
            benchmark_authors=bm_authors,
            benchmark_tags=bm_tags,
            task_id=task_id,
            task_version_hash=task_version_hash,
            seed=seed,
            split=split,
            abstract_description=abstract_description,
            first_observation_text=_extract_first_observation_text(trajectory),
            recommended_max_steps=recommended_max_steps,
            extra_info=extra_info,
            eval_function_ref=eval_function_ref,
        )


class TaskEvalRecord(TypedBaseModel):
    """Complete evaluation record for one agent-task episode.

    Compatible with Project ATLAS TaskEvalRecord schema. The on-disk format is plain
    JSON (JSONL), consumable by any framework without a cube-harness dependency.

    Field groups follow the ATLAS design:
        task        — what was evaluated
        agent       — who evaluated it
        outcome     — how it went (reward, errors)
        trajectory  — execution summary (steps, tokens, time)
        provenance  — when, where, with what framework version
    """

    # Nested descriptors
    task: TaskInfo
    agent: AgentInfo

    # Outcome
    success: bool = Field(description="True when final reward > 0.")
    reward: float = Field(description="Final reward from the last EnvironmentOutput.")
    reward_breakdown: dict = Field(
        default_factory=dict,
        description="Full reward_info dict from the trajectory (may include sub-goal scores, done flag, etc.).",
    )
    error_type: str | None = Field(
        default=None,
        description=(
            "Exception class name if any step raised an error (e.g. 'TimeoutError', 'ValueError'). "
            "None for clean episodes (even if reward=0)."
        ),
    )

    # Trajectory summary
    n_steps: int = Field(description="Total trajectory steps (agent + env combined).")
    n_agent_steps: int = Field(description="Agent steps (LLM decision turns).")
    n_env_steps: int = Field(description="Environment steps (tool executions).")
    wall_time_s: float | None = Field(default=None, description="Episode wall-clock duration in seconds.")
    usage: UsageSummary = Field(
        default_factory=UsageSummary,
        description="Aggregated LLM token usage and cost for the episode.",
    )

    # Provenance
    run_id: str = Field(description="Unique run identifier: '{exp_name}_{trajectory_id}'.")
    trajectory_id: str = Field(description="Trajectory ID as stored on disk.")
    timestamp: float = Field(description="Episode start time as Unix timestamp.")
    framework_version: str = Field(description="cube-harness version.")

    @classmethod
    def from_trajectory(
        cls,
        trajectory: Trajectory,
        agent_info: AgentInfo,
        task_info: TaskInfo,
        exp_name: str = "",
    ) -> "TaskEvalRecord":
        """Assemble a TaskEvalRecord from a completed trajectory and pre-built info objects."""
        harness_version = _get_package_version("cube-harness") or "unknown"
        last_env = trajectory.last_env_step()
        reward = last_env.reward
        stats = trajectory.summary_stats or {}

        wall_time_s: float | None = None
        if trajectory.start_time is not None and trajectory.end_time is not None:
            wall_time_s = trajectory.end_time - trajectory.start_time

        return cls(
            task=task_info,
            agent=agent_info,
            success=reward > 0,
            reward=reward,
            reward_breakdown=dict(trajectory.reward_info),
            error_type=_extract_error_type(trajectory),
            n_steps=len(trajectory.steps),
            n_agent_steps=stats.get("n_agent_steps", trajectory.n_agent_steps),
            n_env_steps=stats.get("n_env_steps", trajectory.n_env_steps),
            wall_time_s=wall_time_s,
            usage=UsageSummary.from_summary_stats(stats),
            run_id=f"{exp_name}_{trajectory.id}" if exp_name else trajectory.id,
            trajectory_id=trajectory.id,
            timestamp=trajectory.start_time or 0.0,
            framework_version=harness_version,
        )


class EvalLog(TypedBaseModel):
    """Collection of TaskEvalRecords with JSONL serialization.

    The JSONL format (one JSON object per line) is the Atlas-compatible wire format.
    Each line is a self-contained TaskEvalRecord — no schema header, no envelope.
    Other frameworks read/write the same format without any cube-harness dependency.

    For large experiments, prefer append_record() (streaming) over save_jsonl()
    (in-memory) to avoid materializing all records at once.
    """

    records: list[TaskEvalRecord] = Field(default_factory=list)

    def save_jsonl(self, path: Path) -> None:
        """Write all records to a JSONL file (one JSON object per line)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for record in self.records:
                f.write(record.model_dump_json() + "\n")
        logger.info(f"Saved {len(self.records)} eval records to {path}")

    @classmethod
    def load_jsonl(cls, path: Path) -> "EvalLog":
        """Load all records from a JSONL file."""
        path = Path(path)
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(TaskEvalRecord.model_validate_json(line))
        return cls(records=records)

    @staticmethod
    def append_record(record: TaskEvalRecord, path: Path) -> None:
        """Append a single record to a JSONL file (streaming mode).

        Suitable for writing records immediately after each episode without holding
        the full log in memory. Not safe for concurrent multi-process writes without
        external file locking.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(record.model_dump_json() + "\n")
