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
_BLOCKED_PATTERNS = [
    re.compile(r"\bsys\.exit\b"),
    re.compile(r"\bos\._exit\b"),
    re.compile(r"\bos\.system\b"),
    re.compile(r"\bsubprocess\b"),
    re.compile(r"\bos\.popen\b"),
    re.compile(r"\bos\.exec\w*\b"),
    re.compile(r"\bos\.spawn\w*\b"),
    re.compile(r"\bos\.kill\b"),
    re.compile(r"\bshutil\.rmtree\b"),
    re.compile(r"\bos\.remove\b"),
    re.compile(r"\bos\.unlink\b"),
]


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
        """Execute Python code inside SandboxFusion and return combined output."""
        self._python_call_count += 1

        stripped = code.strip()
        if not stripped:
            self._last_python_output = "[no output]"
            return self._last_python_output

        for pattern in _BLOCKED_PATTERNS:
            if pattern.search(stripped):
                self._last_python_output = f"Blocked: code contains forbidden pattern '{pattern.pattern}'"
                return self._last_python_output

        if RunCodeRequest is None or run_code is None:
            self._last_python_output = "[execution error: sandbox-fusion is not installed]"
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
            run_result = self._obj_get(response, "run_result", {}) or {}
            status = str(self._obj_get(response, "status", ""))
            message = str(self._obj_get(response, "message", "") or "")
            stdout = (self._obj_get(run_result, "stdout", "") or "").rstrip()
            stderr = (self._obj_get(run_result, "stderr", "") or "").rstrip()
            is_timeout = "timeout" in status.lower() or "timeout" in message.lower()

            parts: list[str] = []
            if stdout:
                parts.append(stdout)
            if stderr:
                parts.append(f"[stderr]\\n{stderr}")
            if is_timeout:
                parts.append("[execution timed out]")
            if not parts:
                parts.append("[no output]")

            self._last_python_output = "\n".join(parts)
            return self._last_python_output
        except Exception as exc:  # pragma: no cover - defensive for runtime tool use
            self._last_python_output = f"[execution error: {exc}]"
            return self._last_python_output

    @tool_action
    def MathAnswer(self, answer: str) -> str:  # noqa: N802
        """Submit the final answer in LaTeX format (for example: ``\\boxed{42}``)."""
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
