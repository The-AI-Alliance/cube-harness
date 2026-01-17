import logging
from typing import Any

from agentlab2.action_spaces.browser_action_space import BrowserActionSpace
from agentlab2.core import ActionSchema, ActionSubset, Observation, Task
from agentlab2.tools.playwright import SyncPlaywrightTool

logger = logging.getLogger(__name__)


class Mind2WebTask(Task):
    validate_per_step: bool = True
    supported_actions: ActionSubset = (
        BrowserActionSpace.browser_press_key,
        BrowserActionSpace.browser_type,
        BrowserActionSpace.browser_click,
        BrowserActionSpace.browser_select_option,
        BrowserActionSpace.browser_hover,
    )
    _tool: SyncPlaywrightTool

    def __init__(
        self,
        id: str,
        task_desc: str,
        website: str,
        domain: str,
        actions_data: list[dict[str, Any]],
        base_url: str,
        episode_max_time: int = 1000000,
        max_turns: int = 20,
    ) -> None:
        self.id = id
        self.task_desc = task_desc
        self.website = website
        self.domain = domain
        self.actions_data = actions_data
        self.base_url = base_url
        self.episode_max_time = episode_max_time
        self.max_turns = max_turns
        self.current_step = 0
        self.ground_truth_actions = self._parse_ground_truth()

    def _parse_ground_truth(self) -> list[dict[str, Any]]:
        parsed = []
        for action in self.actions_data:
            op_data = action["operation"]
            op_type = op_data["op"]

            if op_type in ("HOVER", "ENTER"):
                op_type = "CLICK"

            parsed.append(
                {
                    "type": op_type,
                    "value": op_data.get("value", ""),
                    "target_element": action.get("pos_candidates", []),
                }
            )
        return parsed

    @property
    def initial_url(self) -> str:
        return f"{self.base_url}/{self.id}_0.html"

    def setup(self, tool: SyncPlaywrightTool) -> tuple[Observation, dict]:
        self._tool = tool
        self.current_step = 0
        logger.info(f"Setting up Mind2Web task {self.id}")

        self._tool.goto(self.initial_url)

        obs = Observation.from_text(self.task_desc)
        obs += self._tool.page_obs()

        return obs, {
            "task_id": self.id,
            "task_url": self.initial_url,
            "task_desc": self.task_desc,
            "website": self.website,
            "domain": self.domain,
            "total_steps": len(self.ground_truth_actions),
        }

    def validate_task(self, *args) -> tuple[float, dict]:
        if self.current_step >= len(self.ground_truth_actions):
            return 1.0, {"done": True, "step": self.current_step, "reason": "All steps completed"}

        next_html_file = f"{self.id}_{self.current_step + 1}.html"
        next_url = f"{self.base_url}/{next_html_file}"

        try:
            self._tool.goto(next_url)
            self.current_step += 1

            if self.current_step >= len(self.ground_truth_actions):
                return 1.0, {"done": True, "step": self.current_step, "reason": "Task completed"}
            else:
                return 0.0, {"done": False, "step": self.current_step, "reason": "Step completed, continuing"}
        except Exception as e:
            logger.warning(f"Failed to load next step HTML: {e}")
            return 0.0, {"done": True, "step": self.current_step, "reason": f"Failed: {e}"}

    def filter_actions(self, actions: list[ActionSchema]) -> list[ActionSchema]:
        supported_action_names = {action.__name__ for action in self.supported_actions}
        filtered = [a for a in actions if a.name in supported_action_names]
        logger.info(f"Chosen {len(filtered)} out of {len(actions)} actions for Mind2Web task.")
        return filtered

    def finished(self) -> bool:
        return self.current_step >= len(self.ground_truth_actions)
