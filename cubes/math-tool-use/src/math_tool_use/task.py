from typing import Any

from cube.benchmark import RuntimeContext
from cube.container import ContainerBackend
from cube.core import Observation
from cube.task import Task, TaskConfig
from math_tool_use.tool import MathToolUseTool, MathToolUseToolConfig


class SolveMathToolUseTask(Task):
    """Solve a math problem via one Python call, then submit final LaTeX answer with MathAnswer."""

    @property
    def _expected(self) -> int:
        return self.metadata.extra_info["expected"]

    def reset(self) -> tuple[Observation, dict[str, Any]]:
        self.tool.reset()
        ei = self.metadata.extra_info
        question = ei["question"]
        return Observation.from_text(question), {"question": question, "expected": self._expected}

    def evaluate(self, obs: Observation) -> tuple[float, dict[str, Any]]:
        assert isinstance(self.tool, MathToolUseTool)

        submitted = self.tool.last_answer
        if submitted is None:
            return 0.0, {
                "status": "in_progress",
                "expected": self._expected,
                "python_call_count": self.tool.python_call_count,
                "answer_call_count": self.tool.answer_call_count,
            }

        parsed_answer = self.tool.parse_integer_from_latex(submitted)
        format_ok = self.tool.is_latex_formatted(submitted)
        workflow_ok = (
            self.tool.python_call_count == 1
            and self.tool.answer_call_count == 1
            and self.tool.python_called_before_answer
        )
        correct = parsed_answer == self._expected
        success = bool(correct and format_ok and workflow_ok)

        return (1.0 if success else 0.0), {
            "submitted": submitted,
            "parsed_answer": parsed_answer,
            "expected": self._expected,
            "format_ok": format_ok,
            "workflow_ok": workflow_ok,
            "correct": correct,
            "python_call_count": self.tool.python_call_count,
            "answer_call_count": self.tool.answer_call_count,
            "python_called_before_answer": self.tool.python_called_before_answer,
            "last_python_output": self.tool.last_python_output,
        }

    def finished(self, obs: Observation) -> bool:
        assert isinstance(self.tool, MathToolUseTool)
        return self.tool.last_answer is not None


class MathToolUseTaskConfig(TaskConfig):
    """Serializable configuration that produces SolveMathToolUseTask."""

    def make(
        self,
        runtime_context: RuntimeContext | None = None,
        container_backend: ContainerBackend | None = None,
    ) -> SolveMathToolUseTask:
        
        # Import here to avoid circular import (benchmark imports task)
        from math_tool_use.benchmark import MathToolUseBenchmark
        task_metadata = MathToolUseBenchmark.task_metadata[self.task_id]
        tool_cfg = self.tool_config or MathToolUseToolConfig()
        
        return SolveMathToolUseTask(
            metadata=task_metadata,
            tool_config=tool_cfg,
            runtime_context=runtime_context,
            container_backend=container_backend,
        )
