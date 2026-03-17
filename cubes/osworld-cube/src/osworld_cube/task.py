"""
OSWorldTask — CUBE task for a single OSWorld desktop-automation episode.

    task = OSWorldTask(metadata=..., tool_config=ComputerConfig(...), vm_backend=LocalQEMUVMBackend(...))
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

from PIL import Image
from pydantic import PrivateAttr

from cube.benchmark import RuntimeContext  # noqa: F401 — triggers OSWorldTask.model_rebuild()
from cube.core import ActionSchema, Observation
from cube.task import Task
from cube.vm import VM, VMBackend, VMConfig

from cube_computer_tool.axtree import linearize_accessibility_tree, tag_screenshot

from osworld_cube.vm_backend.evaluator import Evaluator
from osworld_cube.vm_backend.setup_controller import SetupController

if TYPE_CHECKING:
    from cube_computer_tool.computer import ComputerBase

logger = logging.getLogger(__name__)


class OSWorldTask(Task):
    """
    A single OSWorld desktop-automation task running inside a VM.

    OSWorld tasks are loaded from JSON files in the OSWorld repository.
    Each task specifies: a natural-language instruction, a VM snapshot
    to restore, setup scripts, and an evaluator configuration.

    Reference: https://github.com/xlang-ai/OSWorld

    Pydantic fields (all inherited from cube.task.Task except use_som and vm_backend):
        metadata:      TaskMetadata  — required; OSWorld-specific fields go in
                                       metadata.extra_info (see below)
        tool_config:   ToolConfig    — required; pass ComputerConfig(...)
        vm_backend:    VMBackend | None — optional; HOW to provision the VM.
                                          Pass LocalQEMUVMBackend(...) for local QEMU.
                                          If None, the tool must have a VM attached
                                          externally via computer.attach_vm().
        validate_per_step: bool      — inherited; default False
        accept_agent_stop: bool      — inherited; default True

    Fields stored in metadata.extra_info:
        domain        (str)   — e.g. "chrome", "os", "libreoffice"
        snapshot      (str)   — VM snapshot name, default "init_state"
        config        (list)  — setup scripts to run before task starts
        evaluator     (dict)  — evaluation function + expected results
        related_apps  (list)  — applications involved in the task

    Task instruction:
        metadata.abstract_description  — used as the agent's goal text
    """

    vm_backend: VMBackend | None = None
    """HOW to provision the VM. If None, a pre-launched VM must be attached via
    ComputerBase.attach_vm() before reset() is called."""

    use_som: bool = False
    """If True, annotate screenshot with numbered bounding boxes (Set-of-Marks)
    and replace axtree with an indexed element table before returning obs."""

    _vm: VM | None = PrivateAttr(default=None)
    _current_task_config: dict | None = PrivateAttr(default=None)
    _action_history: list = PrivateAttr(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        """Create the Computer tool without a VM — VM is deferred to reset()."""
        self._tool = self.tool_config.make(container=None, vm=None)

    @property
    def _computer(self) -> "ComputerBase":
        """Return self.tool cast to ComputerBase for type-checker satisfaction."""
        return self.tool  # type: ignore[return-value]

    def _os_type(self) -> str:
        """Return the OS type string ('ubuntu' or 'windows') for axtree processing."""
        raw = self.metadata.extra_info.get("os_type", "Ubuntu")
        return raw.lower()

    def _ensure_vm(self) -> None:
        """Launch the VM if a vm_backend is configured and no VM is running yet."""
        if self._vm is not None:
            return
        if self.vm_backend is None:
            return

        snapshot = self.metadata.extra_info.get("snapshot", "init_state")
        vm_config = VMConfig(snapshot_name=snapshot)
        logger.info("Launching VM via %s", type(self.vm_backend).__name__)
        self._vm = self.vm_backend.launch(vm_config)
        self._computer.attach_vm(self._vm)

    def _get_vm_ports(self) -> tuple[int, int, int]:
        """Return (chromium_port, vlc_port, server_port) from the live VM.

        Attempts to read port attributes from the VM handle. Falls back to
        defaults (9222, 8080, 5000) if the VM does not expose these attributes
        (e.g. a custom VM backend).
        """
        vm = self._vm
        chromium_port: int = getattr(vm, "chromium_port", 9222)
        vlc_port: int = getattr(vm, "vlc_port", 8080)
        server_port: int = getattr(vm, "server_port", 5000)
        return chromium_port, vlc_port, server_port

    def _setup_task(self, task_config: dict) -> Observation:
        """Restore VM snapshot, run setup scripts, wait, return initial observation.

        Called from reset(). Uses SetupController for OSWorld-specific task
        configuration scripts.
        """
        logger.info("Setting up task: %s", task_config.get("id", "unknown"))
        logger.info("Instruction: %s", task_config.get("instruction", ""))

        if self._vm is not None:
            snapshot = task_config.get("snapshot", "init_state")
            self._vm.restore_snapshot(snapshot)

        setup_steps = task_config.get("config") or []
        if setup_steps:
            chromium_port, vlc_port, _ = self._get_vm_ports()
            task_cache_dir = str(Path(self._computer.config.cache_dir) / task_config.get("id", "task"))
            Path(task_cache_dir).mkdir(parents=True, exist_ok=True)
            setup_ctrl = SetupController(
                guest=self._computer._guest,
                chromium_port=chromium_port,
                vlc_port=vlc_port,
                cache_dir=task_cache_dir,
                screen_width=1920,
                screen_height=1080,
            )
            setup_ctrl.setup(setup_steps)

        logger.info("Waiting 60s for VM to stabilise...")
        time.sleep(60)

        self._computer._is_done = False
        self._action_history = []
        self._current_task_config = task_config
        return self._computer.get_observation()

    def _evaluate_task(self) -> float:
        """Run the OSWorld evaluator and return reward ∈ [0.0, 1.0]."""
        if self._current_task_config is None:
            logger.error("_evaluate_task() called before reset()")
            return 0.0

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
        try:
            reward = evaluator.evaluate(self._current_task_config, self._action_history)
            logger.info("Task evaluation result: %f", reward)
            return reward
        except Exception as exc:
            logger.error("Evaluation failed: %s", exc)
            return 0.0

    def reset(self) -> tuple[Observation, dict]:
        """
        Restore the VM snapshot, run setup scripts, and return the initial obs.

        Steps:
          1. Launch VM if not yet running (via vm_backend)
          2. Build task_config dict from metadata.extra_info
          3. Restore VM snapshot, run setup scripts, wait for stabilisation
          4. Post-process the observation (SoM or linearize)
          5. Prepend task instruction as text observation
          6. Return (obs, info)
        """
        self._ensure_vm()
        extra = self.metadata.extra_info

        task_config = {
            "id": self.metadata.id,
            "instruction": self.metadata.abstract_description,
            "config": extra.get("config", []),
            "evaluator": extra.get("evaluator", {}),
            "snapshot": extra.get("snapshot", "init_state"),
            "related_apps": extra.get("related_apps", []),
        }

        logger.info("Resetting OSWorldTask %s (domain=%s)", self.metadata.id, extra.get("domain", "unknown"))

        obs = self._setup_task(task_config)
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
        """
        Call the task evaluator and return (reward, info).

        reward ∈ [0.0, 1.0]:  1.0 = task fully completed.
        Partial credit is preserved (not rounded to binary).
        """
        evaluator_cfg = self.metadata.extra_info.get("evaluator", {})

        if not evaluator_cfg:
            logger.warning("Task %s: no evaluator configured, returning 0.0", self.metadata.id)
            return 0.0, {"error": "no_evaluator"}

        eval_func = evaluator_cfg.get("func", "unknown")
        logger.debug("Evaluating task %s with evaluator: %s", self.metadata.id, eval_func)

        reward = self._evaluate_task()
        logger.info("Task %s evaluation: reward=%f, evaluator=%s", self.metadata.id, reward, eval_func)
        return reward, {
            "evaluator": eval_func,
            "expected": evaluator_cfg.get("expected", {}),
        }

    def finished(self, obs: Observation) -> bool:
        """Return True if the task has reached a terminal state (done() or fail() called)."""
        return self._computer._is_done

    def obs_postprocess(self, obs: Observation) -> Observation:
        """Post-process raw observation before returning to the agent."""
        if self.use_som:
            return self._postprocess_som(obs)
        return self._postprocess_linearize(obs)

    def _postprocess_linearize(self, obs: Observation) -> Observation:
        """Replace raw axtree XML with a linearized tab-separated table."""
        platform = self._os_type()
        new_contents = []
        for content in obs.contents:
            if content.name == "accessibility_tree":
                try:
                    axtree_txt = linearize_accessibility_tree(content.data, platform=platform)
                    new_contents.append(content.model_copy(update={"data": axtree_txt, "name": "axtree_txt"}))
                except Exception as e:
                    logger.warning("Failed to linearize accessibility tree: %s", e)
                    new_contents.append(content)
            else:
                new_contents.append(content)
        return obs.model_copy(update={"contents": new_contents})

    def _postprocess_som(self, obs: Observation) -> Observation:
        """Annotate screenshot with numbered bounding boxes (Set-of-Marks).

        Falls back to _postprocess_linearize if screenshot or axtree are missing,
        or if the annotation fails.
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
            tagged_img.load()  # force load before BytesIO goes out of scope

            new_contents = []
            for content in obs.contents:
                if content.name == "screenshot" and isinstance(content.data, Image.Image):
                    new_contents.append(content.model_copy(update={"data": tagged_img}))
                elif content.name == "accessibility_tree":
                    new_contents.append(content.model_copy(update={"data": element_list, "name": "som_elements"}))
                else:
                    new_contents.append(content)
            return obs.model_copy(update={"contents": new_contents})

        except Exception as e:
            logger.warning("Failed to apply SoM annotation: %s", e)
            return self._postprocess_linearize(obs)

    def filter_actions(self, actions: list[ActionSchema]) -> list[ActionSchema]:
        """Optionally whitelist a subset of Computer actions for this task.

        Default: allow all actions.
        """
        return actions

    def close(self) -> None:
        """Clean up task resources: stop tool then stop VM."""
        logger.info("Closing OSWorldTask: %s", self.metadata.id)
        super().close()  # calls self.tool.close()
        if self._vm is not None:
            self._vm.stop()
            self._vm = None
