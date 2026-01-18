import json
import logging
import re
from typing import Any

from agentlab2.action_spaces.browser_action_space import BrowserActionSpace
from agentlab2.core import Action, ActionSchema, ActionSubset, Observation, Task
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

    # Action type mappings between agent actions and Mind2Web operations
    # Mind2Web uses: CLICK, TYPE, SELECT, HOVER, PRESS
    # Agent uses: browser_click, browser_type, browser_select_option, browser_hover, browser_press_key
    AGENT_TO_MIND2WEB: dict[str, str] = {
        "browser_click": "CLICK",
        "browser_type": "TYPE",
        "browser_select_option": "SELECT",
        "browser_hover": "HOVER",
        "browser_press_key": "PRESS",
    }
    MIND2WEB_TO_AGENT: dict[str, str] = {v: k for k, v in AGENT_TO_MIND2WEB.items()}

    def __init__(
        self,
        id: str,
        task_desc: str,
        website: str,
        domain: str,
        actions_data: list[dict[str, Any]],
        base_url: str,
        action_reprs: list[str] | None = None,
        episode_max_time: int = 1000000,
        max_turns: int = 20,
        binary_scoring: bool = True,
    ) -> None:
        self.id = id
        self.task_desc = task_desc
        self.website = website
        self.domain = domain
        self.actions_data = actions_data
        self.action_reprs = action_reprs or []  # Human-readable GT action strings from dataset
        self.base_url = base_url
        self.episode_max_time = episode_max_time
        self.max_turns = max_turns
        self.binary_scoring = binary_scoring
        self.current_step = 0
        self.steps_correct = 0
        self.ground_truth_actions = self._parse_ground_truth()

    def _parse_ground_truth(self) -> list[dict[str, Any]]:
        """Parse ground truth actions from Mind2Web format."""
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
                    "pos_candidates": action.get("pos_candidates", []),
                }
            )
        return parsed

    @property
    def initial_url(self) -> str:
        return f"{self.base_url}/{self.id}_0.html"

    def setup(self, tool: SyncPlaywrightTool) -> tuple[Observation, dict]:
        self._tool = tool
        self.current_step = 0
        self.steps_correct = 0
        logger.info(f"Setting up Mind2Web task {self.id}")

        self._tool.goto(self.initial_url)

        obs = self._tool.page_obs()
        obs += Observation.from_text(self._build_task_context())

        return obs, {
            "task_id": self.id,
            "task_url": self.initial_url,
            "task_desc": self.task_desc,
            "website": self.website,
            "domain": self.domain,
            "total_steps": len(self.ground_truth_actions),
        }

    def teardown(self) -> None:
        """Reset task state after completion."""
        self.current_step = 0
        self.steps_correct = 0

    def _map_action_type(self, agent_action: Action) -> str | None:
        """Map agent action name to Mind2Web operation type."""
        return self.AGENT_TO_MIND2WEB.get(agent_action.name)

    def _map_mind2web_action_type(self, mind2web_action_type: str) -> str | None:
        """Map Mind2Web operation type to agent action name."""
        return self.MIND2WEB_TO_AGENT.get(mind2web_action_type)

    def _extract_element_attributes(self, agent_action: Action) -> dict[str, Any]:
        """Extract element attributes from agent action arguments."""
        args = agent_action.arguments
        attributes = {}

        if "selector" in args:
            selector = args["selector"]
            attributes["selector"] = selector

            # Parse common selector patterns to extract attributes
            if "name=" in selector:
                if name_match := re.search(r'name=["\']?([^"\'>\]]+)', selector):
                    attributes["name"] = name_match.group(1)

            if "id=" in selector:
                if id_match := re.search(r'id=["\']?([^"\'>\]]+)', selector):
                    attributes["id"] = id_match.group(1)

        return attributes

    def _match_element(self, agent_attributes: dict[str, Any], pos_candidates: list[dict[str, Any]]) -> bool:
        """Check if agent action targets match any positive candidate element."""
        if not pos_candidates:
            return False

        for candidate in pos_candidates:
            # Parse candidate attributes (stored as JSON string)
            if "attributes" in candidate:
                try:
                    cand_attrs = json.loads(candidate["attributes"])
                except (json.JSONDecodeError, TypeError):
                    cand_attrs = {}
            else:
                cand_attrs = {}

            # Match on name attribute if available
            if "name" in agent_attributes and "name" in cand_attrs:
                if agent_attributes["name"] == cand_attrs["name"]:
                    return True

            # Match on id attribute if available
            if "id" in agent_attributes and "id" in cand_attrs:
                if agent_attributes["id"] == cand_attrs["id"]:
                    return True

        return False

    def _evaluate_action(self, agent_action: Action, gt_action: dict[str, Any]) -> bool:
        """Compare agent action against ground truth action."""
        # Map agent action to Mind2Web operation type
        agent_op_type = self._map_action_type(agent_action)
        gt_op_type = gt_action["type"]

        # Check if action types match
        if agent_op_type != gt_op_type:
            logger.debug(f"Action type mismatch: agent={agent_op_type}, expected={gt_op_type}")
            return False

        # For TYPE and SELECT actions, check if value matches
        if gt_op_type in ("TYPE", "SELECT"):
            agent_value = agent_action.arguments.get("text") or agent_action.arguments.get("value", "")
            gt_value = gt_action.get("value", "")

            if agent_value.lower().strip() != gt_value.lower().strip():
                logger.debug(f"Value mismatch: agent={agent_value}, expected={gt_value}")
                return False

        # Check if target element matches
        agent_attrs = self._extract_element_attributes(agent_action)
        if not self._match_element(agent_attrs, gt_action["pos_candidates"]):
            logger.debug(f"Element mismatch: agent_attrs={agent_attrs}")
            return False

        return True

    def _calculate_reward(self) -> float:
        """Calculate reward based on scoring mode and steps completed."""
        if self.binary_scoring:
            return 1.0 if self.steps_correct == len(self.ground_truth_actions) else 0.0
        return self.steps_correct / len(self.ground_truth_actions) if self.ground_truth_actions else 0.0

    def validate_task(self, obs: Observation, actions: list[Action] | None = None) -> tuple[float, dict]:
        actions = actions or []

        # Check if we've completed all steps
        if self.current_step >= len(self.ground_truth_actions):
            return self._calculate_reward(), {
                "done": True,
                "step": self.current_step,
                "steps_correct": self.steps_correct,
                "total_steps": len(self.ground_truth_actions),
                "reason": "All steps completed",
            }

        # Get ground truth for current step
        gt_action = self.ground_truth_actions[self.current_step]

        # Evaluate the agent's last action if available
        action_correct = False
        if actions:
            agent_action = actions[-1]
            action_correct = self._evaluate_action(agent_action, gt_action)
            if action_correct:
                self.steps_correct += 1
                logger.info(f"Step {self.current_step} CORRECT: {agent_action.name} matched ground truth")
            else:
                logger.info(
                    f"Step {self.current_step} INCORRECT: {agent_action.name} did not match ground truth {gt_action['type']}"
                )

        # Advance to next step
        self.current_step += 1

        # Load next HTML snapshot
        next_html_file = f"{self.id}_{self.current_step}.html"
        next_url = f"{self.base_url}/{next_html_file}"

        try:
            self._tool.goto(next_url)
        except Exception as e:
            logger.warning(f"Failed to load next step HTML: {e}")
            # Task ends early due to missing HTML
            return self._calculate_reward(), {
                "done": True,
                "step": self.current_step,
                "steps_correct": self.steps_correct,
                "total_steps": len(self.ground_truth_actions),
                "reason": f"Failed to load HTML: {e}",
            }

        # Check if we've completed all steps after advancing
        if self.current_step >= len(self.ground_truth_actions):
            return self._calculate_reward(), {
                "done": True,
                "step": self.current_step,
                "steps_correct": self.steps_correct,
                "total_steps": len(self.ground_truth_actions),
                "reason": "Task completed",
            }

        # Continue to next step
        return 0.0, {
            "done": False,
            "step": self.current_step,
            "steps_correct": self.steps_correct,
            "total_steps": len(self.ground_truth_actions),
            "action_correct": action_correct,
            "reason": "Step completed, continuing",
        }

    def filter_actions(self, actions: list[ActionSchema]) -> list[ActionSchema]:
        supported_action_names = {action.__name__ for action in self.supported_actions}
        filtered = [a for a in actions if a.name in supported_action_names]
        logger.info(f"Chosen {len(filtered)} out of {len(actions)} actions for Mind2Web task.")
        return filtered

    def finished(self) -> bool:
        return self.current_step >= len(self.ground_truth_actions)

    def obs_postprocess(self, obs: Observation) -> Observation:
        """Build observation with fresh page content (validate_task navigates to new HTML snapshot)."""
        # Fetch fresh page content since validate_task() already navigated to the next HTML
        # I dislike this logic so much
        fresh_obs = self._tool.page_obs()
        return fresh_obs + Observation.from_text(self._build_task_context())

    def _format_previous_actions(self) -> str:
        """Format GT previous actions in our tool format for inclusion in observation."""
        if self.current_step == 0 or not self.action_reprs:
            return "Previous actions: None"

        lines = ["Previous actions:"]
        for i in range(min(self.current_step, len(self.action_reprs))):
            action_repr = self.action_reprs[i]
            # Convert Mind2Web format (e.g., "-> CLICK", "-> TYPE:") to agent format
            for mind2web_op, agent_op in self.MIND2WEB_TO_AGENT.items():
                # Handle actions with values (TYPE:, SELECT:) and without (CLICK, HOVER, PRESS)
                action_repr = action_repr.replace(f"-> {mind2web_op}:", f"-> {agent_op}:")
                action_repr = action_repr.replace(f"-> {mind2web_op}", f"-> {agent_op}")
            lines.append(f"  Step {i + 1}: {action_repr}")
        return "\n".join(lines)

    def _build_task_context(self) -> str:
        """Build the task context prompt for the agent."""
        previous_actions_text = self._format_previous_actions()
        return f"""
Based on the webpage above, try to complete the following task:
Task: {self.task_desc}
{previous_actions_text}
What should be the next action?"""
