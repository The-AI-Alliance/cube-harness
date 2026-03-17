import json
import logging
from pathlib import Path
from typing import Any

from cube.benchmark import RuntimeContext
from cube.container import ContainerBackend
from cube.core import Observation
from cube.task import Task, TaskConfig, TaskMetadata
from webarena_verified.api.webarena_verified import WebArenaVerified
from webarena_verified.types.config import WebArenaVerifiedConfig
from webarena_verified.types.eval import EvalStatus, NetworkTrace, TaskEvalResult
from webarena_verified.types.task import WebArenaVerifiedTask as WAVTask

from cube_harness.tools.playwright import SyncPlaywrightTool
from cube_harness.tools.toolbox import Toolbox
from webarena_verified_cube.tool import SubmitResponseTool, WebArenaToolConfig

logger = logging.getLogger(__name__)


class WebArenaVerifiedTask(Task):
    wav_task: WAVTask
    wav_config: WebArenaVerifiedConfig

    @property
    def _playwright_tool(self) -> SyncPlaywrightTool:
        assert isinstance(self.tool, Toolbox)
        tool = self.tool.find_tool(SyncPlaywrightTool)
        assert tool is not None
        return tool

    @property
    def _submit_tool(self) -> SubmitResponseTool:
        assert isinstance(self.tool, Toolbox)
        tool = self.tool.find_tool(SubmitResponseTool)
        assert tool is not None
        return tool

    def reset(self) -> tuple[Observation, dict[str, Any]]:
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
        self._playwright_tool.close()  # HAR is saved at context close
        har_path = Path(self._playwright_tool.config.context_kwargs["record_har_path"])
        network_trace = NetworkTrace.from_har(har_path)
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
            tool_config=self.tool_config or WebArenaToolConfig(),
            wav_task=self.wav_task,
            wav_config=self.wav_config,
        )
