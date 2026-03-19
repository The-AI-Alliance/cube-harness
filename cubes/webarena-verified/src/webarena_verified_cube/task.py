import logging
from pathlib import Path
from typing import Any

from cube.benchmark import RuntimeContext
from cube.container import ContainerBackend
from cube.core import Observation
from cube.task import Task, TaskConfig, TaskMetadata
from pydantic import PrivateAttr
from webarena_verified.api.webarena_verified import WebArenaVerified
from webarena_verified.types.config import WebArenaVerifiedConfig
from webarena_verified.types.eval import EvalStatus, NetworkTrace, TaskEvalResult
from webarena_verified.types.task import WebArenaVerifiedTask as WAVTask

from cube_browser_tool import SyncPlaywrightTool
from cube_harness.tools.toolbox import Toolbox, ToolboxConfig
from webarena_verified_cube.tool import HarPlaywrightConfig, SubmitResponseConfig, SubmitResponseTool

logger = logging.getLogger(__name__)


class WebArenaVerifiedTask(Task):
    wav_task: WAVTask
    wav_config: WebArenaVerifiedConfig

    _playwright_closed: bool = PrivateAttr(default=False)

    @property
    def _playwright_tool(self) -> SyncPlaywrightTool:
        if not isinstance(self.tool, Toolbox):
            raise TypeError(f"Expected Toolbox, got {type(self.tool).__name__}")
        tool = self.tool.find_tool(SyncPlaywrightTool)
        if tool is None:
            raise RuntimeError("SyncPlaywrightTool not found in Toolbox")
        return tool

    @property
    def _submit_tool(self) -> SubmitResponseTool:
        if not isinstance(self.tool, Toolbox):
            raise TypeError(f"Expected Toolbox, got {type(self.tool).__name__}")
        tool = self.tool.find_tool(SubmitResponseTool)
        if tool is None:
            raise RuntimeError("SubmitResponseTool not found in Toolbox")
        return tool

    def reset(self) -> tuple[Observation, dict[str, Any]]:
        self._playwright_closed = False
        self.tool.reset()
        start_url = self.wav_config.render_url(self.wav_task.start_urls[0], list(self.wav_task.sites), strict=False)
        self._playwright_tool.goto(start_url)
        obs = Observation.from_text(self.wav_task.intent) + self._playwright_tool.page_obs()
        info = {
            "task_id": self.wav_task.task_id,
            "sites": [s.value for s in self.wav_task.sites],
            "expected_action": self.wav_task.expected_action,
        }
        return obs, info

    def evaluate(self, obs: Observation) -> tuple[float, dict[str, Any]]:
        submitted = self._submit_tool.get_submitted_response()
        if submitted is None:
            return 0.0, {"eval_status": EvalStatus.FAILURE, "evaluators_results": []}
        # Close the browser context to flush the HAR to disk, then read it.
        # The framework will call Toolbox.close() afterwards; SyncPlaywrightTool.stop()
        # is safe to call twice (errors are logged but not re-raised).
        if not self._playwright_closed:
            self._playwright_tool.close()
            self._playwright_closed = True
        har_path = Path(self._playwright_tool.config.har_path)
        network_trace = NetworkTrace.from_har(har_path)
        har_path.unlink(missing_ok=True)
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
        return self._submit_tool.get_submitted_response() is not None


class WebArenaVerifiedTaskConfig(TaskConfig):
    task_metadata: TaskMetadata
    wav_task: WAVTask
    wav_config: WebArenaVerifiedConfig

    def make(
        self,
        runtime_context: RuntimeContext | None = None,
        container_backend: ContainerBackend | None = None,
    ) -> WebArenaVerifiedTask:
        _ = runtime_context, container_backend
        return WebArenaVerifiedTask(
            metadata=self.task_metadata,
            tool_config=self.tool_config or ToolboxConfig(tool_configs=[HarPlaywrightConfig(), SubmitResponseConfig()]),
            wav_task=self.wav_task,
            wav_config=self.wav_config,
        )
