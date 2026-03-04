from typing import Any

from cube.benchmark import RuntimeContext
from cube.container import ContainerBackend
from cube.core import Observation
from cube.task import Task, TaskConfig, TaskMetadata
from arithmetic_cube.tool import ArithmeticTool, ArithmeticToolConfig


class SolveArithmeticTask(Task):
    """Task: solve a math problem by calling submit_answer() once with the correct integer."""

    @property
    def _expected(self) -> int:
        return self.metadata.extra_info["expected"]

    def reset(self) -> tuple[Observation, dict[str, Any]]:
        self.tool.reset()
        ei = self.metadata.extra_info
        question = f"What is {ei['a']} {ei['op']} {ei['b']}?"
        return Observation.from_text(question), {"question": question, "expected": self._expected}

    def evaluate(self, obs: Observation) -> tuple[float, dict[str, Any]]:
        assert isinstance(self.tool, ArithmeticTool)
        answer = self.tool.last_answer
        correct = answer == self._expected
        return (1.0 if correct else 0.0), {"answer": answer, "expected": self._expected, "correct": correct}

    def finished(self, obs: Observation) -> bool:
        assert isinstance(self.tool, ArithmeticTool)
        return self.tool.last_answer is not None


class ArithmeticTaskConfig(TaskConfig):
    """Serializable configuration that produces a SolveArithmeticTask."""

    def make(
        self,
        runtime_context: RuntimeContext | None = None,
        container_backend: ContainerBackend | None = None,
    ) -> SolveArithmeticTask:
        # Import here to avoid circular import (benchmark imports task)
        from arithmetic_cube.benchmark import ArithmeticBenchmark

        task_metadata: TaskMetadata = ArithmeticBenchmark.task_metadata[self.task_id]
        tool_cfg = self.tool_config or ArithmeticToolConfig()
        return SolveArithmeticTask(
            metadata=task_metadata,
            tool_config=tool_cfg,
            runtime_context=runtime_context,
            container_backend=container_backend,
        )
