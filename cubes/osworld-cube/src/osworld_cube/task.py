"""
OSWorldTask — CUBE task for a single OSWorld desktop-automation episode.

    task = OSWorldTask(metadata=..., tool_config=ComputerConfig(...))
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
from typing import TYPE_CHECKING

from PIL import Image

from cube.benchmark import RuntimeContext  # noqa: F401 — triggers OSWorldTask.model_rebuild()
from cube.core import ActionSchema, Observation
from cube.task import Task, TaskMetadata

if TYPE_CHECKING:
    from osworld_cube.computer import ComputerBase

logger = logging.getLogger(__name__)


class OSWorldTask(Task):
    """
    A single OSWorld desktop-automation task running inside a VM.

    OSWorld tasks are loaded from JSON files in the OSWorld repository.
    Each task specifies: a natural-language instruction, a VM snapshot
    to restore, setup scripts, and an evaluator configuration.

    Reference: https://github.com/xlang-ai/OSWorld

    Pydantic fields (all inherited from cube.task.Task except use_som):
        metadata:      TaskMetadata  — required; OSWorld-specific fields go in
                                       metadata.extra_info (see below)
        tool_config:   ToolConfig    — required; pass ComputerConfig(...)
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

    use_som: bool = False
    """If True, annotate screenshot with numbered bounding boxes (Set-of-Marks)
    and replace axtree with an indexed element table before returning obs."""

    @property
    def _computer(self) -> "Computer":
        """Return self.tool cast to Computer for type-checker satisfaction."""
        return self.tool  # type: ignore[return-value]


    def reset(self) -> tuple[Observation, dict]:
        """
        Restore the VM snapshot, run setup scripts, and return the initial obs.

        self.tool is already available here (set by Task.model_post_init via
        tool_config.make()). No tool argument needed.

        Steps:
          1. Reset _is_done on the tool
          2. Build task_config dict from metadata.extra_info
          3. Call self._computer.setup_task(task_config) → raw Observation
             (blocks ~60s for snapshot restore + stabilisation)
          4. Post-process the observation (SoM or linearize)
          5. Prepend task instruction as text observation
          6. Return (obs, info)
        """
        extra = self.metadata.extra_info

        self._computer._is_done = False

        task_config = {
            "id": self.metadata.id,
            "instruction": self.metadata.abstract_description,
            "config": extra.get("config", []),
            "evaluator": extra.get("evaluator", {}),
            "snapshot": extra.get("snapshot", "init_state"),
            "related_apps": extra.get("related_apps", []),
        }

        logger.info(
            f"Resetting OSWorldTask {self.metadata.id} "
            f"(domain={extra.get('domain', 'unknown')})"
        )

        obs = self._computer.setup_task(task_config)
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
        Call desktop_env's built-in evaluator and return (reward, info).

        reward ∈ [0.0, 1.0]:  1.0 = task fully completed.
        Partial credit is preserved (not rounded to binary).

        The obs argument is accepted to satisfy the cube.task.Task interface
        but is unused — OSWorld evaluation inspects VM state directly via
        desktop_env.evaluate(), not the agent's latest observation.
        """
        evaluator = self.metadata.extra_info.get("evaluator", {})

        if not evaluator:
            logger.warning(
                f"Task {self.metadata.id}: no evaluator configured, returning 0.0"
            )
            return 0.0, {"error": "no_evaluator"}

        eval_func = evaluator.get("func", "unknown")
        logger.debug(
            f"Evaluating task {self.metadata.id} with evaluator: {eval_func}"
        )

        reward = self._computer.evaluate_task()
        logger.info(
            f"Task {self.metadata.id} evaluation: reward={reward}, evaluator={eval_func}"
        )
        return reward, {
            "evaluator": eval_func,
            "expected": evaluator.get("expected", {}),
        }


    def finished(self, obs: Observation) -> bool:
        """
        Return True if the task has reached a terminal state.

        cube.task.Task.step() calls this after every action.
        If True, step() triggers evaluate() and sets done=True.

        _is_done is set on the Computer tool when the agent calls done() or
        fail() @tool_action. This acts as a fallback alongside the inherited
        STOP_ACTION (final_step) handling in Task.step().
        """
        return self._computer._is_done

    def obs_postprocess(self, obs: Observation) -> Observation:
        """Post-process raw observation before returning to the agent."""
        if self.use_som:
            return self._postprocess_som(obs)
        return self._postprocess_linearize(obs)

    def _postprocess_linearize(self, obs: Observation) -> Observation:
        """Replace raw axtree XML with a linearized tab-separated table.

        The platform string is read from ComputerConfig.os_type.
        """
        from osworld_cube.axtree import linearize_accessibility_tree

        platform = self._computer.config.os_type.lower()
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
        """Annotate screenshot with numbered bounding boxes and replace axtree
        with an indexed element table (Set-of-Marks).

        Falls back to _postprocess_linearize if screenshot or axtree are missing,
        or if the annotation fails.
        """
        from osworld_cube.axtree import tag_screenshot

        platform = self._computer.config.os_type.lower()

        screenshot_content = None
        axtree_content = None
        for content in obs.contents:
            if content.name == "screenshot" and isinstance(content.data, Image.Image):
                screenshot_content = content
            elif content.name == "accessibility_tree":
                axtree_content = content

        if screenshot_content is None or axtree_content is None:
            logger.warning(
                "SoM requires both screenshot and accessibility_tree; "
                "falling back to linearize."
            )
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
                    new_contents.append(
                        content.model_copy(
                            update={"data": element_list, "name": "som_elements"}
                        )
                    )
                else:
                    new_contents.append(content)
            return obs.model_copy(update={"contents": new_contents})

        except Exception as e:
            logger.warning(f"Failed to apply SoM annotation: {e}")
            return self._postprocess_linearize(obs)

    def filter_actions(self, actions: list[ActionSchema]) -> list[ActionSchema]:
        """Optionally whitelist a subset of Computer actions for this task.

        Default: allow all actions. Override in task subclasses for
        domain-specific action restriction (e.g. disable hotkey for web tasks).
        """
        return actions

    def close(self) -> None:
        """Clean up task resources.

        Calls super().close() which calls self.tool.close() to shut down the VM.
        """
        logger.info(f"Closing OSWorldTask: {self.metadata.id}")
        super().close()  # calls self.tool.close()
