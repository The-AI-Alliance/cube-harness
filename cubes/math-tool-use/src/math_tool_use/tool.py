import re
from typing import Any

from cube.container import Container
from cube.tool import Tool, ToolConfig, tool_action

try:
    from sandbox_fusion import RunCodeRequest, run_code
except ImportError:  # pragma: no cover - optional dependency at runtime
    RunCodeRequest = None  # type: ignore[assignment]
    run_code = None  # type: ignore[assignment]

_INTEGER_PATTERN = re.compile(r"^-?\d+$")


class MathToolUseToolConfig(ToolConfig):
    sandbox_endpoint: str | None = None
    sandbox_language: str = "python"
    sandbox_compile_timeout: float = 10.0
    sandbox_run_timeout: float = 10.0
    sandbox_client_timeout: float = 15.0
    sandbox_max_attempts: int = 2

    def make(self, container: Container | None = None) -> "MathToolUseTool":
        return MathToolUseTool(config=self)


class MathToolUseTool(Tool):
    def __init__(self, config: MathToolUseToolConfig) -> None:
        self.config = config
        self._python_call_count: int = 0
        self._answer_call_count: int = 0
        self._python_called_before_answer: bool = True
        self._submitted_answer: str | None = None
        self._last_python_output: str | None = None

    def reset(self) -> None:
        self._python_call_count = 0
        self._answer_call_count = 0
        self._python_called_before_answer = True
        self._submitted_answer = None
        self._last_python_output = None

    @tool_action
    def run_python_code(self, code: str) -> str:
        """Execute Python code exactly once inside SandboxFusion to compute or verify the result.

        Args:
            code: Python code to execute.
        """
        self._python_call_count += 1
        if self._python_call_count > 1:
            return "run_python_code may only be called once."

        stripped = code.strip()
        if not stripped:
            self._last_python_output = "Python error: empty code."
            return self._last_python_output

        if RunCodeRequest is None or run_code is None:
            self._last_python_output = (
                "Python error: sandbox-fusion is not installed. Install with `pip install sandbox-fusion`."
            )
            return self._last_python_output

        try:
            request = RunCodeRequest(
                code=stripped,
                language=self.config.sandbox_language,
                compile_timeout=self.config.sandbox_compile_timeout,
                run_timeout=self.config.sandbox_run_timeout,
            )
            kwargs: dict[str, Any] = {
                "max_attempts": self.config.sandbox_max_attempts,
                "client_timeout": self.config.sandbox_client_timeout,
            }
            if self.config.sandbox_endpoint:
                kwargs["endpoint"] = self.config.sandbox_endpoint

            response = run_code(request, **kwargs)
            status = self._obj_get(response, "status", "")
            message = self._obj_get(response, "message", "")
            run_result = self._obj_get(response, "run_result", {}) or {}
            run_status = self._obj_get(run_result, "status", "")
            return_code = self._obj_get(run_result, "return_code", None)
            stdout = (self._obj_get(run_result, "stdout", "") or "").strip()
            stderr = (self._obj_get(run_result, "stderr", "") or "").strip()

            if stdout:
                self._last_python_output = f"Python output: {stdout}"
                return self._last_python_output

            if status != "Success" or run_status != "Finished" or return_code not in (0, None):
                detail = stderr or message or "Sandbox execution failed without details."
                self._last_python_output = f"Python error: {detail}"
                return self._last_python_output

            self._last_python_output = "Python code executed with no stdout."
            return self._last_python_output
        except Exception as exc:  # pragma: no cover - defensive for runtime tool use
            self._last_python_output = f"Python error: {exc.__class__.__name__}: {exc}"
            return self._last_python_output

    @tool_action
    def MathAnswer(self, answer: str) -> str:  # noqa: N802
        """Submit the final answer in LaTeX format (for example: ``\\boxed{42}``).

        Args:
            answer: Final LaTeX-formatted answer.
        """
        self._answer_call_count += 1
        if self._answer_call_count > 1:
            return "MathAnswer may only be called once."

        if self._python_call_count == 0:
            self._python_called_before_answer = False

        self._submitted_answer = answer.strip()
        return "Final answer submitted."

    @staticmethod
    def parse_integer_from_latex(answer: str) -> int | None:
        text = answer.strip()
        if text.startswith("$") and text.endswith("$") and len(text) >= 2:
            text = text[1:-1].strip()

        boxed_match = re.fullmatch(r"\\boxed\{(.+)\}", text)
        if boxed_match:
            text = boxed_match.group(1).strip()

        if _INTEGER_PATTERN.fullmatch(text):
            return int(text)
        return None

    @staticmethod
    def _obj_get(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @staticmethod
    def is_latex_formatted(answer: str) -> bool:
        text = answer.strip()
        return ("\\boxed" in text) or ("$" in text) or ("\\" in text)

    @property
    def python_call_count(self) -> int:
        return self._python_call_count

    @property
    def answer_call_count(self) -> int:
        return self._answer_call_count

    @property
    def python_called_before_answer(self) -> bool:
        return self._python_called_before_answer

    @property
    def last_answer(self) -> str | None:
        return self._submitted_answer

    @property
    def last_python_output(self) -> str | None:
        return self._last_python_output
