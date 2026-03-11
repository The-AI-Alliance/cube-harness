"""OSWorld task implementation for desktop automation.

Ported from agentlab2.benchmarks.osworld.task. Desktop_env dependency removed;
tools are accessed via the AbstractComputerTool protocol.
"""

import io
import logging
from typing import TYPE_CHECKING, List, Optional

from PIL import Image

from agentlab2.core import ActionSchema, Observation, Task
from osworld_cube.osworld_axtree import linearize_accessibility_tree, tag_screenshot

if TYPE_CHECKING:
    from cube.tools.computer import AbstractComputerTool

logger = logging.getLogger(__name__)


class OSWorldTask(Task):
    """Represents a single OSWorld task.

    Manages the lifecycle of one desktop automation episode: VM setup,
    observation post-processing (linearize or SoM), and evaluation.

    Attributes:
        id: Unique task identifier.
        desc: Human-readable task description.
        domain: Task domain (``"chrome"``, ``"os"``, ``"libreoffice"``, etc.).
        instruction: Natural language instruction shown to the agent.
        snapshot: VM snapshot name to restore (default: ``"init_state"``).
        related_apps: Applications involved in the task.
        config: Setup commands to run before the task starts.
        evaluator: Evaluation configuration.
        max_turns: Maximum agent turns allowed.
        use_som: Use Set-of-Marks screenshot annotation.
    """

    validate_per_step: bool = False

    def __init__(
        self,
        id: str,
        desc: str,
        domain: str = "general",
        instruction: str = "",
        snapshot: str = "init_state",
        related_apps: Optional[List[str]] = None,
        config: Optional[List[dict]] = None,
        evaluator: Optional[dict] = None,
        max_turns: int = 15,
        use_som: bool = False,
    ) -> None:
        self.id = id
        self.desc = desc
        self.domain = domain
        self.instruction = instruction
        self.snapshot = snapshot
        self.related_apps = related_apps or []
        self.config = config or []
        self.evaluator = evaluator or {}
        self.max_turns = max_turns
        self.use_som = use_som
        self._tool: "AbstractComputerTool | None" = None
        self._is_done: bool = False

    def setup(self, tool: "AbstractComputerTool") -> tuple[Observation, dict]:
        """Set up the task: restore VM, run setup commands, return initial obs.

        Args:
            tool: Any AbstractComputerTool implementation.

        Returns:
            (initial observation with task instruction, info dict)
        """
        self._tool = tool
        self._is_done = False
        logger.info("Setting up OSWorld task: %s (domain=%s)", self.id, self.domain)

        task_config = {
            "id": self.id,
            "instruction": self.instruction,
            "config": self.config,
            "evaluator": self.evaluator,
            "snapshot": self.snapshot,
            "related_apps": self.related_apps,
        }

        obs = self._tool.setup_task(task_config)
        obs = self.obs_postprocess(obs)

        goal_obs = Observation.from_text(f"Task: {self.instruction or self.desc}")
        obs = goal_obs + obs

        info = {
            "task_id": self.id,
            "task_desc": self.desc,
            "task_domain": self.domain,
            "task_snapshot": self.snapshot,
            "task_related_apps": self.related_apps,
        }
        return obs, info

    def validate_task(self, obs: Observation) -> tuple[float, dict]:
        """Evaluate task completion via the tool.

        Args:
            obs: Current observation (not used; evaluation is VM-based).

        Returns:
            (reward 0.0–1.0, info dict)
        """
        if not self._tool:
            return 0.0, {"done": False, "error": "no_tool"}
        if not self.evaluator:
            return 0.0, {"done": False, "error": "no_evaluator"}

        eval_func = self.evaluator.get("func", "unknown")
        try:
            reward = self._tool.evaluate_task()
            done = reward > 0.0 or self._is_done
            return reward, {"done": done, "evaluator": eval_func}
        except Exception as exc:
            logger.error("Task validation failed: %s", exc)
            return 0.0, {"done": False, "error": str(exc)}

    def filter_actions(self, actions: List[ActionSchema]) -> List[ActionSchema]:
        return actions

    def obs_postprocess(self, obs: Observation) -> Observation:
        """Post-process observation (linearize or SoM)."""
        if self.use_som:
            return self._postprocess_som(obs)
        return self._postprocess_linearize(obs)

    def _postprocess_linearize(self, obs: Observation) -> Observation:
        new_contents = []
        for content in obs.contents:
            if content.name == "accessibility_tree":
                try:
                    axtree_txt = linearize_accessibility_tree(content.data, platform="ubuntu")
                    new_contents.append(
                        content.model_copy(update={"data": axtree_txt, "name": "axtree_txt"})
                    )
                except Exception as exc:
                    logger.warning("Failed to linearize accessibility tree: %s", exc)
                    new_contents.append(content)
            else:
                new_contents.append(content)
        return obs.model_copy(update={"contents": new_contents})

    def _postprocess_som(self, obs: Observation) -> Observation:
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
                screenshot_bytes, axtree_content.data, platform="ubuntu"
            )
            # Store marks on tool if it supports update_marks (e.g. for pyautogui mode)
            if self._tool is not None and hasattr(self._tool, "update_marks"):
                self._tool.update_marks(marks)  # type: ignore[attr-defined]

            tagged_img = Image.open(io.BytesIO(tagged_screenshot_bytes))
            tagged_img.load()

            new_contents = []
            for content in obs.contents:
                if content.name == "screenshot" and isinstance(content.data, Image.Image):
                    new_contents.append(content.model_copy(update={"data": tagged_img}))
                elif content.name == "accessibility_tree":
                    new_contents.append(
                        content.model_copy(update={"data": element_list, "name": "som_elements"})
                    )
                else:
                    new_contents.append(content)
            return obs.model_copy(update={"contents": new_contents})

        except Exception as exc:
            logger.warning("Failed to apply SoM annotation: %s", exc)
            return self._postprocess_linearize(obs)

    def finished(self) -> bool:
        if self._tool is not None:
            return getattr(self._tool, "_is_done", False)
        return self._is_done

    def mark_done(self, success: bool = False) -> None:
        self._is_done = True

    def teardown(self) -> None:
        logger.info("Tearing down OSWorld task: %s", self.id)
