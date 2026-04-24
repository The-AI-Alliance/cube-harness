from typing import Any

from cube.benchmark import RuntimeContext
from cube.container import ContainerBackend
from cube.core import Observation
from cube.task import EvalInfo, Task, TaskConfig
from math_tool_use.tool import MathToolUseTool, MathToolUseToolConfig
import math_verify

_BONUS_ON_CORRECT_WITH_PYTHON = 0.1
_PENALTY_ON_INCORRECT_WITHOUT_PYTHON = 0.1
_MAX_ABS_SHAPING = 0.2


def _verify_answer_status(
    prediction: str,
    gold: str,
    *,
    strict: bool = True,
    max_prediction_length: int = 1000,
) -> str:
    boxed_start = prediction.rfind("\\boxed{")
    if boxed_start < 0:
        return "no_answer"

    boxed_prediction = prediction[boxed_start:]
    if "\\boxed{}" in boxed_prediction:
        return "unparsable"
    if len(boxed_prediction) > max_prediction_length:
        return "unparsable"

    try:
        gold_parsed = math_verify.parse(gold)
        boxed_prediction_parsed = math_verify.parse(boxed_prediction)
        if not boxed_prediction_parsed:
            return "unparsable"
        equivalent = math_verify.verify(gold_parsed, boxed_prediction_parsed, strict=strict, timeout_seconds=1)
        return "correct" if equivalent else "wrong"
    except Exception:
        return "unparsable"


def _python_tool_shaping(answer_status: str, python_call_count: int) -> float:
    shaping = 0.0
    if answer_status == "correct" and python_call_count >= 1:
        shaping += _BONUS_ON_CORRECT_WITH_PYTHON
    if answer_status in ("wrong", "unparsable") and python_call_count == 0:
        shaping -= _PENALTY_ON_INCORRECT_WITHOUT_PYTHON
    return max(-_MAX_ABS_SHAPING, min(_MAX_ABS_SHAPING, shaping))


class SolveMathToolUseTask(Task):
    """Solve a math problem via Python and submit final LaTeX answer with MathAnswer."""

    accept_agent_stop: bool = False  # agent should not be allowed to stop before submitting an answer
    validate_per_step: bool = False  # only validate at the end based on submitted answer, not per step

    @property
    def _expected(self) -> str:
        return self.metadata.extra_info["expected"]

    def reset(self) -> tuple[Observation, dict[str, Any]]:
        self.tool.reset()
        ei = self.metadata.extra_info
        question = ei["question"]
        return Observation.from_text(question), {"question": question, "expected": self._expected}

    def evaluate(self, obs: Observation | None = None) -> tuple[float, EvalInfo | dict[str, Any]]:
        assert isinstance(self.tool, MathToolUseTool)

        submitted = self.tool.last_answer
        if submitted is None:
            return 0.0, EvalInfo(
                success=False,
                no_answer=True,
                no_error=True,
                status="no_answer",
                extra={
                    "answer_status": "no_answer",
                    "expected": self._expected,
                    "python_call_count": self.tool.python_call_count,
                    "answer_call_count": self.tool.answer_call_count,
                    "python_called_before_answer": self.tool.python_called_before_answer,
                    "last_python_output": self.tool.last_python_output,
                },
            )

        answer_status = _verify_answer_status(submitted, self._expected, strict=True)
        base_reward = 1.0 if answer_status == "correct" else 0.0
        shaping = _python_tool_shaping(answer_status, self.tool.python_call_count)
        reward = base_reward + shaping

        return reward, EvalInfo(
            success=answer_status == "correct",
            no_answer=answer_status == "no_answer",
            no_error=answer_status != "unparsable",
            status=answer_status,
            extra={
                "answer_status": answer_status,
                "submitted": submitted,
                "expected": self._expected,
                "base_reward": base_reward,
                "python_tool_shaping": shaping,
                "python_call_count": self.tool.python_call_count,
                "answer_call_count": self.tool.answer_call_count,
                "python_called_before_answer": self.tool.python_called_before_answer,
                "last_python_output": self.tool.last_python_output,
            },
        )

    def finished(self, obs: Observation | None = None) -> bool:
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
            accept_agent_stop=False,
        )
