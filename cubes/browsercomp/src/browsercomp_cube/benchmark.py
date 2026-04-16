"""BrowseCompBenchmark: 1,266 web information-retrieval tasks."""

import csv
import io
import logging
from importlib.resources import files
from typing import ClassVar, Generator

from cube.benchmark import Benchmark, BenchmarkMetadata
from cube.task import TaskConfig, TaskMetadata

from browsercomp_cube.crypto import decrypt
from browsercomp_cube.task import BrowseCompTaskConfig

logger = logging.getLogger(__name__)

_CSV_PATH = files("browsercomp_cube.data") / "browse_comp_test_set.csv"


def _load_task_metadata() -> dict[str, TaskMetadata]:
    """Read and decrypt the BrowseComp CSV at import time.

    Each row's problem and answer are XOR-decrypted using the per-row canary
    string. Returns a dict mapping task IDs to TaskMetadata.
    """
    metadata: dict[str, TaskMetadata] = {}
    text = _CSV_PATH.read_text(encoding="utf-8")
    for idx, row in enumerate(csv.DictReader(io.StringIO(text))):
        canary = row["canary"]
        problem = decrypt(row["problem"], canary)
        answer = decrypt(row["answer"], canary)
        topic = row.get("problem_topic", "")
        task_id = f"browsecomp-{idx:04d}"
        metadata[task_id] = TaskMetadata(
            id=task_id,
            abstract_description=topic,
            recommended_max_steps=50,
            extra_info={"problem": problem, "answer": answer, "topic": topic},
        )
    return metadata


class BrowseCompBenchmark(Benchmark):
    """BrowseComp benchmark: 1,266 hard web information-retrieval tasks."""

    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="browsercomp-cube",
        version="0.1.0",
        description="BrowseComp benchmark — hard web information retrieval requiring multi-step browsing",
        num_tasks=1266,
        tags=["web-search", "information-retrieval"],
    )
    task_metadata: ClassVar[dict[str, TaskMetadata]] = _load_task_metadata()
    task_config_class: ClassVar[type[TaskConfig]] = BrowseCompTaskConfig

    scorer_model: str = "gpt-5.4-mini"

    def _setup(self) -> None:
        pass

    def close(self) -> None:
        pass

    def get_task_configs(self) -> Generator[BrowseCompTaskConfig, None, None]:
        for tm in self.task_metadata.values():
            yield BrowseCompTaskConfig(
                task_id=tm.id,
                tool_config=self.default_tool_config,
                scorer_model=self.scorer_model,
            )
