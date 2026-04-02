import logging
from typing import Any, overload

from cube.benchmark import RuntimeContext
from cube.container import ContainerBackend
from cube.core import Observation
from cube.task import Task, TaskConfig
from cube.tools.browser import BrowserTool
from pydantic import PrivateAttr
from webarena_verified.api.webarena_verified import WebArenaVerified
from webarena_verified.types.config import WebArenaVerifiedConfig
from webarena_verified.types.eval import EvalStatus, TaskEvalResult
from webarena_verified.types.task import WebArenaVerifiedTask as WAVTask

from cube.tool import Toolbox, ToolboxConfig
from webarena_verified_cube.tool import HarPlaywrightConfig, SubmitResponseConfig, SubmitResponseTool, WAVBrowserTool

logger = logging.getLogger(__name__)


@overload
def _render_url(config: WebArenaVerifiedConfig, url: str, sites: list) -> str: ...
@overload
def _render_url(config: WebArenaVerifiedConfig, url: list[str], sites: list) -> list[str]: ...
def _render_url(config: WebArenaVerifiedConfig, url: str | list[str], sites: list) -> str | list[str]:
    return config.render_url(url, sites, strict=False)


class WebArenaVerifiedTask(Task):
    wav_task: WAVTask
    wav_config: WebArenaVerifiedConfig

    _playwright_closed: bool = PrivateAttr(default=False)

    @property
    def _browser_tool(self) -> WAVBrowserTool:
        if not isinstance(self.tool, Toolbox):
            raise TypeError(f"Expected Toolbox, got {type(self.tool).__name__}")

        tool = self.tool.find_tool(BrowserTool)
        if not isinstance(tool, WAVBrowserTool):
            raise RuntimeError("BrowserTool not found in Toolbox or missing network_trace()")
        return tool

    @property
    def _submit_tool(self) -> SubmitResponseTool:
        if not isinstance(self.tool, Toolbox):
            raise TypeError(f"Expected Toolbox, got {type(self.tool).__name__}")
        tool = self.tool.find_tool(SubmitResponseTool)
        if not isinstance(tool, SubmitResponseTool):
            raise RuntimeError("SubmitResponseTool not found in Toolbox")
        return tool

    def reset(self) -> tuple[Observation, dict[str, Any]]:
        """Reset the task by reinitializing the browser tool and navigating to the task's start URL.

        Returns an observation combining the task intent text and the initial page state,
        along with task metadata (task_id, sites, expected_action).
        """
        self._playwright_closed = False
        self.tool.reset()
        start_url = _render_url(self.wav_config, self.wav_task.start_urls[0], list(self.wav_task.sites))
        self._browser_tool.goto(start_url)
        obs = Observation.from_text(self.wav_task.intent) + self._browser_tool.page_obs()
        info = {
            "task_id": self.wav_task.task_id,
            "sites": [s.value for s in self.wav_task.sites],
            "expected_action": self.wav_task.expected_action,
        }
        return obs, info

    def evaluate(self, obs: Observation) -> tuple[float, dict[str, Any]]:
        """Evaluate the agent's submitted response against the WebArena verified evaluators.

        Closes the browser context to flush the HAR file to disk, reads the network trace,
        then calls the WebArenaVerified API to score the response. Returns 0.0 immediately
        if no response was submitted.

        Returns the score and a dict with eval_status and per-evaluator results.
        """
        submitted = self._submit_tool.get_submitted_response()
        if submitted is None:
            return 0.0, {"eval_status": EvalStatus.FAILURE, "evaluators_results": []}
        if not self._playwright_closed:
            self._browser_tool.close()
            self._playwright_closed = True
        network_trace = self._browser_tool.network_trace()
        wav = WebArenaVerified(config=self.wav_config)
        result: TaskEvalResult = wav.evaluate_task(
            task_id=self.wav_task.task_id,
            agent_response=submitted.model_dump(),
            network_trace=network_trace,
        )
        return result.score, {
            "eval_status": result.status,
            "evaluators_results": [r.model_dump() for r in result.evaluators_results],
        }

    def finished(self, obs: Observation) -> bool:
        """Return True once the agent has submitted a response via the SubmitResponseTool."""
        return self._submit_tool.get_submitted_response() is not None


class WebArenaVerifiedTaskConfig(TaskConfig):
    wav_task: WAVTask
    wav_config: WebArenaVerifiedConfig

    def make(
        self,
        runtime_context: RuntimeContext | None = None,
        container_backend: ContainerBackend | None = None,
    ) -> WebArenaVerifiedTask:
        _ = runtime_context, container_backend
        from webarena_verified_cube.benchmark import WebArenaVerifiedBenchmark

        metadata = WebArenaVerifiedBenchmark.task_metadata[self.task_id]
        return WebArenaVerifiedTask(
            metadata=metadata,
            tool_config=self.tool_config or ToolboxConfig(tool_configs=[HarPlaywrightConfig(), SubmitResponseConfig()]),
            wav_task=self.wav_task,
            wav_config=self.wav_config,
        )
