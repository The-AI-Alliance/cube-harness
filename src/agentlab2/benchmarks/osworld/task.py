"""OSWorld task implementation for desktop automation.

This module provides the OSWorldTask class that manages the lifecycle of a single
OSWorld task, including VM setup, observation building, and evaluation.
"""

import io
import logging
from typing import TYPE_CHECKING, List, Optional

from PIL import Image

from agentlab2.benchmarks.osworld.osworld_axtree import linearize_accessibility_tree, tag_screenshot
from agentlab2.core import ActionSchema, Observation, Task

if TYPE_CHECKING:
    from agentlab2.tools.computer import Computer

logger = logging.getLogger(__name__)


class OSWorldTask(Task):
    """Represents a single OSWorld task.

    OSWorld tasks are desktop-based tasks that run inside Docker containers
    or VMs, where agents interact via mouse/keyboard actions and observe
    via screenshots and accessibility trees.

    Reference: https://github.com/xlang-ai/OSWorld

    Attributes:
        id: Unique task identifier (UUID in original OSWorld)
        desc: Human-readable task description
        domain: Task domain (e.g., "chrome", "os", "libreoffice", "vscode")
        instruction: The natural language instruction for the agent
        snapshot: VM snapshot name to restore before task (default: "init_state")
        related_apps: List of applications involved in the task
        config: Setup scripts to run before task starts
        evaluator: Evaluation configuration with function name and expected results
        validate_per_step: Whether to validate after each step (default: False)
        max_turns: Maximum number of agent turns allowed
        use_som: Whether to use Set-of-Marks screenshot annotation (default: False)
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
        self._tool: "Computer | None" = None
        self._is_done: bool = False

    def setup(self, tool: "Computer") -> tuple[Observation, dict]:
        """Set up the OSWorld task.

        This method:
        1. Stores reference to the tool for later use
        2. Calls tool.setup_task() which restores VM and waits for stabilization
        3. Gets initial observation with screenshot and accessibility tree
        4. Prepends task instruction to observation

        Args:
            tool: The Computer tool instance for interacting with the VM

        Returns:
            Tuple of (initial observation, task info dict)
        """
        self._tool = tool
        self._is_done = False
        logger.info(f"Setting up OSWorld task: {self.id} (domain={self.domain})")

        # Build task config for desktop_env
        task_config = {
            "id": self.id,
            "instruction": self.instruction,
            "config": self.config,
            "evaluator": self.evaluator,
            "snapshot": self.snapshot,
            "related_apps": self.related_apps,
        }

        # Setup VM (includes 60s wait and initial observation)
        obs = self._tool.setup_task(task_config)
        obs = self.obs_postprocess(obs)

        # Prepend instruction as text observation
        goal = self.instruction or self.desc
        goal_obs = Observation.from_text(f"Task: {goal}")
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
        """Validate if the task has been completed successfully.

        This calls desktop_env's built-in evaluation which handles various
        evaluation methods depending on the task:
        - File existence/content checks
        - Application state verification
        - Screenshot comparison
        - Custom evaluation scripts

        Args:
            obs: Current observation (not used, evaluation is VM-based)

        Returns:
            Tuple of (reward 0.0-1.0, info dict with 'done' key)
        """
        if not self._tool:
            logger.warning(f"No tool available for task {self.id}")
            return 0.0, {"done": False, "error": "no_tool"}

        if not self.evaluator:
            logger.warning(f"No evaluator configured for task {self.id}")
            return 0.0, {"done": False, "error": "no_evaluator"}

        eval_func = self.evaluator.get("func", "unknown")
        logger.debug(f"Validating task {self.id} with evaluator: {eval_func}")

        try:
            # Call desktop_env's evaluate() method
            reward = self._tool.evaluate_task()
            done = reward > 0.0 or self._is_done

            logger.info(
                f"Task {self.id} validation: reward={reward}, done={done}, evaluator={eval_func}"
            )

            return reward, {
                "done": done,
                "evaluator": eval_func,
                "expected": self.evaluator.get("expected", {}),
            }
        except Exception as e:
            logger.error(f"Task validation failed: {e}")
            return 0.0, {"done": False, "error": str(e)}

    def filter_actions(self, actions: List[ActionSchema]) -> List[ActionSchema]:
        """Filter available actions for this task.

        By default, all Computer actions are allowed.
        Override in subclasses or configure via task JSON if needed.

        Args:
            actions: Full list of available actions from the tool

        Returns:
            Filtered list of actions allowed for this task
        """
        # By default, allow all actions
        return actions

    def obs_postprocess(self, obs: Observation) -> Observation:
        """Post-process observation before returning to agent.

        Dispatches to SoM or linearize mode depending on self.use_som.

        Args:
            obs: Raw observation from the tool

        Returns:
            Processed observation
        """
        if self.use_som:
            return self._postprocess_som(obs)
        return self._postprocess_linearize(obs)

    def _postprocess_linearize(self, obs: Observation) -> Observation:
        """Replace raw axtree XML with a linearized tab-separated table.

        Args:
            obs: Raw observation containing accessibility_tree content

        Returns:
            Observation with axtree_txt content replacing accessibility_tree
        """
        platform = self._tool.config.os_type.lower() if self._tool else "ubuntu"
        new_contents = []
        for content in obs.contents:
            if content.name == "accessibility_tree":
                try:
                    axtree_txt = linearize_accessibility_tree(content.data, platform=platform)
                    new_contents.append(
                        content.model_copy(update={"data": axtree_txt, "name": "axtree_txt"})
                    )
                except Exception as e:
                    logger.warning(f"Failed to linearize accessibility tree: {e}")
                    new_contents.append(content)
            else:
                new_contents.append(content)
        return obs.model_copy(update={"contents": new_contents})

    def _postprocess_som(self, obs: Observation) -> Observation:
        """Annotate screenshot with numbered bounding boxes and replace axtree with indexed element table.

        Bounding boxes are drawn at the screen coordinates reported by the OS accessibility APIs
        for each visible, interactive element. Falls back to linearize if screenshot or axtree
        are missing, or if annotation fails.

        Args:
            obs: Raw observation containing screenshot and accessibility_tree content

        Returns:
            Observation with annotated screenshot and som_elements text content
        """
        platform = self._tool.config.os_type.lower() if self._tool else "ubuntu"

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
            if self._tool is not None:
                self._tool.update_marks(marks)

            tagged_img = Image.open(io.BytesIO(tagged_screenshot_bytes))
            tagged_img.load()  # force load before BytesIO goes out of scope

            # Rebuild contents in original order, replacing screenshot and axtree in place
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
            logger.warning(f"Failed to apply SoM annotation: {e}")
            return self._postprocess_linearize(obs)

    def finished(self) -> bool:
        """Check if the task is finished.

        Returns:
            True if task has reached a terminal state
        """
        if self._tool is not None:
            return self._tool._is_done
        return self._is_done

    def mark_done(self, success: bool = False) -> None:
        """Mark the task as done.

        Called when agent signals DONE or FAIL, or when max_turns reached.

        Args:
            success: Whether the task was completed successfully
        """
        self._is_done = True
        logger.info(f"Task {self.id} marked as done (success={success})")

    def teardown(self) -> None:
        """Clean up after task completion.

        VM cleanup is handled by the Computer tool's close() method.
        """
        logger.info(f"Tearing down OSWorld task: {self.id}")
        # Cleanup is handled at the tool level
