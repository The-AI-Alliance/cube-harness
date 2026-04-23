"""WAATask — CUBE task for a single WindowsAgentArena desktop-automation episode.

task = WAATask(metadata=..., tool_config=ComputerConfig(...), infra=LocalInfraConfig())
obs, info = task.reset()
while not done:
    action = agent(obs, task.action_set)
    env_out = task.step(action)
    obs, done = env_out.obs, env_out.done
task.close()
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from cube.benchmark import RuntimeContext  # noqa: F401 — triggers WAATask.model_rebuild()
from cube.core import Observation
from cube.resource import InfraConfig, ResourceHandle
from cube.task import Task
from cube_computer_tool.axtree import linearize_accessibility_tree, tag_screenshot
from PIL import Image
from pydantic import PrivateAttr

from waa_cube.azure import WAA_WINDOWS_RESOURCE
from waa_cube.vm_backend.evaluator import Evaluator
from waa_cube.vm_backend.setup_controller import SetupController

if TYPE_CHECKING:
    from cube_computer_tool.computer import ComputerBase

logger = logging.getLogger(__name__)

_POST_SNAPSHOT_SLEEP = 10  # seconds to wait after QMP loadvm before taking obs

# Actual VM resolution after snapshot restore.  The QEMU display adapter is
# initialised at 1920×1080 (required for the Windows accessibility API), but the
# snapshots were captured at 1280×800 so the guest reverts to that on restore.
_VM_SCREEN_WIDTH = 1280
_VM_SCREEN_HEIGHT = 800


def _reformat_axtree(raw: str) -> str:
    """Reformat linearize_accessibility_tree output into the agent-facing table.

    Input columns (from linearize_accessibility_tree):
        tag  name  text  class  description  position (top-left x&y)  size (w&h)

    Output columns:
        index  tag  name  text  x  y  w  h

    Drops class (pywinauto internal) and description (almost always empty).
    Unpacks position/size tuple strings into separate integer columns.
    """
    lines = raw.splitlines()
    if not lines:
        return raw

    out = ["index\ttag\tname\ttext\tx\ty\tw\th"]
    idx = 1
    for line in lines[1:]:  # skip header
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        tag, name, text = parts[0], parts[1], parts[2]
        # parts[3]=class, parts[4]=description — dropped
        pos_str, size_str = parts[5], parts[6]
        try:
            x, y = (int(v.strip()) for v in pos_str.strip("()").split(","))
            w, h = (int(v.strip()) for v in size_str.strip("()").split(","))
        except ValueError:
            continue
        out.append(f"{idx}\t{tag}\t{name}\t{text}\t{x}\t{y}\t{w}\t{h}")
        idx += 1
    return "\n".join(out)


class WAATask(Task):
    """A single WAA desktop-automation task running inside a Windows 11 VM.

    WAA tasks are loaded from JSON files in evaluation_examples_windows/.
    Each task specifies: a natural-language instruction, a named QEMU snapshot
    to restore, setup scripts, and an evaluator configuration.

    Pydantic fields:
        metadata:      TaskMetadata  — extra_info keys: domain, snapshot, config,
                                       evaluator, related_apps
        tool_config:   ToolConfig    — pass ComputerConfig(...)
        infra:         InfraConfig   — InfraConfig used to launch task VMs.
        validate_per_step: bool      — inherited; default False
        accept_agent_stop: bool      — inherited; default True
    """

    infra: InfraConfig | None = None
    """InfraConfig (LocalInfraConfig, AzureInfraConfig, ...)."""

    use_som: bool = False
    """If True, annotate screenshot with numbered bounding boxes (Set-of-Marks)."""

    _resource_handle: ResourceHandle | None = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Create the Computer tool without a VM — VM is deferred to reset()."""
        self._tool = self.tool_config.make(container=None, vm=None)

    @property
    def _computer(self) -> "ComputerBase":
        return self.tool  # type: ignore[return-value]

    def _os_type(self) -> str:
        """WAA always runs Windows 11."""
        return "windows"

    def _ensure_vm(self) -> None:
        """Launch the VM via infra if not already running."""
        if self._resource_handle is not None:
            return
        if self.infra is None:
            raise RuntimeError("WAATask requires an InfraConfig — set infra= when constructing.")

        logger.info("Launching VM via %s", type(self.infra).__name__)
        self._resource_handle = self.infra.launch(WAA_WINDOWS_RESOURCE)
        self._computer.attach_endpoint(self._resource_handle.endpoint)

    def _get_vm_ports(self) -> tuple[int, int, int]:
        """Return (chromium_port, vlc_port, server_port) from the current handle."""
        server_port = 5000
        if self._resource_handle is not None and self._resource_handle.endpoint:
            server_port = urlparse(self._resource_handle.endpoint).port or 5000
        return 9222, 8080, server_port

    def _setup_task(self, task_data: dict) -> Observation:
        """Run setup scripts, wait, return initial observation."""
        logger.info(
            "Setting up WAA task: %s. Instruction: %s",
            task_data.get("id", "unknown"),
            task_data.get("instruction", ""),
        )

        setup_steps = task_data.get("config") or []
        chromium_port, vlc_port, _ = self._get_vm_ports()
        task_cache_dir = str(Path(self._computer.config.cache_dir) / task_data.get("id", "task"))
        Path(task_cache_dir).mkdir(parents=True, exist_ok=True)
        setup_ctrl = SetupController(
            guest=self._computer._guest,
            chromium_port=chromium_port,
            vlc_port=vlc_port,
            cache_dir=task_cache_dir,
            screen_width=_VM_SCREEN_WIDTH,
            screen_height=_VM_SCREEN_HEIGHT,
        )
        # Always wait for Flask agent connectivity before proceeding — the guest
        # agent may need time to start on first boot of a cloud VM.
        reachable = setup_ctrl.setup(setup_steps)
        if not reachable:
            logger.warning("WAA VM guest agent unreachable — observation may fail")

        did_something = self._resource_handle is not None or bool(setup_steps)
        if did_something:
            logger.info("Waiting %ds for VM to stabilise...", _POST_SNAPSHOT_SLEEP)
            time.sleep(_POST_SNAPSHOT_SLEEP)

        return self._computer.get_observation()

    def _evaluate_task(self) -> float:
        """Run the WAA evaluator and return reward ∈ [0.0, 1.0]."""
        if self._computer._guest is None:
            logger.error("_evaluate_task() called with no VM attached")
            return 0.0

        chromium_port, vlc_port, server_port = self._get_vm_ports()
        cache_dir_base = Path(self._computer.config.cache_dir)

        evaluator = Evaluator(
            guest=self._computer._guest,
            cache_dir_base=cache_dir_base,
            chromium_port=chromium_port,
            vlc_port=vlc_port,
            server_port=server_port,
        )
        eval_config = {
            "id": self.metadata.id,
            "evaluator": self.metadata.extra_info.get("evaluator", {}),
        }
        try:
            reward = evaluator.evaluate(eval_config, self._computer._action_history)
            logger.info("WAA task evaluation result: %f", reward)
            return reward
        except Exception as exc:
            logger.error("Evaluation failed: %s", exc)
            return 0.0

    def reset(self) -> tuple[Observation, dict]:
        """Run setup scripts and return the initial obs.

        Steps:
          1. Launch VM if not yet running (via infra)
          2. Build task_data dict from metadata.extra_info
          3. Run setup scripts, wait
          4. Post-process the observation (SoM or linearize axtree)
          5. Prepend task instruction as text observation
          6. Return (obs, info)
        """
        self._ensure_vm()
        extra = self.metadata.extra_info

        task_data = {
            "id": self.metadata.id,
            "instruction": self.metadata.abstract_description,
            "config": extra.get("config", []),
            "evaluator": extra.get("evaluator", {}),
            "snapshot": extra.get("snapshot", "init_state"),
            "related_apps": extra.get("related_apps", []),
        }

        logger.info("Resetting WAATask %s (domain=%s)", self.metadata.id, extra.get("domain", "unknown"))

        obs = self._setup_task(task_data)
        obs = self.obs_postprocess(obs)

        goal_obs = Observation.from_text(f"Task: {self.metadata.abstract_description}")
        obs = goal_obs + obs

        info = {
            "task_id": self.metadata.id,
            "task_domain": extra.get("domain", "unknown"),
            "task_snapshot": extra.get("snapshot", "init_state"),
            "task_related_apps": extra.get("related_apps", []),
        }
        return obs, info

    def evaluate(self, obs: Observation) -> tuple[float, dict]:
        """Call the WAA task evaluator and return (reward, info).

        reward ∈ [0.0, 1.0]:  1.0 = task fully completed.
        """
        evaluator_cfg = self.metadata.extra_info.get("evaluator", {})

        if not evaluator_cfg:
            logger.warning("Task %s: no evaluator configured, returning 0.0", self.metadata.id)
            return 0.0, {"error": "no_evaluator"}

        eval_func = evaluator_cfg.get("func", "unknown")
        logger.debug("Evaluating WAA task %s with evaluator: %s", self.metadata.id, eval_func)

        reward = self._evaluate_task()
        logger.info("WAA task %s evaluation: reward=%f, evaluator=%s", self.metadata.id, reward, eval_func)
        return reward, {
            "evaluator": eval_func,
            "expected": evaluator_cfg.get("expected", {}),
        }

    def finished(self, obs: Observation) -> bool:
        """Return True if the task has reached a terminal state."""
        return self._computer._is_done

    def obs_postprocess(self, obs: Observation) -> Observation:
        """Post-process raw observation before returning to the agent."""
        if self.use_som:
            return self._postprocess_som(obs)
        return self._postprocess_linearize(obs)

    def _postprocess_linearize(self, obs: Observation) -> Observation:
        """Replace raw axtree XML with a clean indexed table for the agent.

        Converts the raw XML to a tab-separated table with columns:
            index  tag  name  text  x  y  w  h

        Drops the class and description columns (noise) and unpacks the
        position/size tuple strings into separate integer columns so the
        agent can compute click centres with: cx = x + w//2, cy = y + h//2.
        """
        platform = self._os_type()
        new_contents = []
        for content in obs.contents:
            if content.name == "accessibility_tree":
                try:
                    raw = linearize_accessibility_tree(content.data, platform=platform)
                    axtree_txt = _reformat_axtree(raw)
                    new_contents.append(content.model_copy(update={"data": axtree_txt, "name": "axtree_txt"}))
                except Exception as exc:
                    logger.warning("Failed to linearize accessibility tree: %s", exc)
                    new_contents.append(content)
            else:
                new_contents.append(content)
        return obs.model_copy(update={"contents": new_contents})

    def _postprocess_som(self, obs: Observation) -> Observation:
        """Annotate screenshot with numbered bounding boxes (Set-of-Marks).

        Falls back to _postprocess_linearize if screenshot or axtree are missing.
        """
        platform = self._os_type()

        screenshot_content = None
        axtree_content = None
        for content in obs.contents:
            if content.name == "screenshot" and isinstance(content.data, Image.Image):
                screenshot_content = content
            elif content.name == "accessibility_tree":
                axtree_content = content

        if screenshot_content is None or axtree_content is None:
            logger.warning("SoM requires both screenshot and accessibility_tree; falling back to linearize.")
            return self._postprocess_linearize(obs)

        try:
            buf = io.BytesIO()
            screenshot_content.data.save(buf, format="PNG")
            screenshot_bytes = buf.getvalue()

            marks, _, tagged_screenshot_bytes, element_list = tag_screenshot(
                screenshot_bytes, axtree_content.data, platform=platform
            )
            self._computer.update_marks(marks)

            tagged_img = Image.open(io.BytesIO(tagged_screenshot_bytes))
            tagged_img.load()

            new_contents = []
            for content in obs.contents:
                if content.name == "screenshot" and isinstance(content.data, Image.Image):
                    new_contents.append(content.model_copy(update={"data": tagged_img}))
                elif content.name == "accessibility_tree":
                    new_contents.append(content.model_copy(update={"data": element_list, "name": "som_elements"}))
                else:
                    new_contents.append(content)
            return obs.model_copy(update={"contents": new_contents})

        except Exception as exc:
            logger.warning("Failed to apply SoM annotation: %s", exc)
            return self._postprocess_linearize(obs)

    def close(self) -> None:
        """Clean up task resources: stop tool and release infra handle."""
        logger.info("Closing WAATask: %s", self.metadata.id)
        super().close()  # calls self.tool.close()
        if self._resource_handle is not None:
            try:
                self._resource_handle.close()
            except Exception as exc:
                logger.warning("Failed to close Azure resource handle: %s", exc)
            self._resource_handle = None
