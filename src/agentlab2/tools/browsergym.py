import logging
from typing import Any, Callable

import numpy as np
from PIL import Image

from agentlab2.core import Action, ActionSchema, Content, Observation
from agentlab2.tool import AbstractTool, ToolConfig
from agentlab2.tools.bgym_core.action.highlevel import HighLevelActionSet
from agentlab2.tools.bgym_core.env import BrowserEnv
from agentlab2.tools.bgym_core.task import OpenEndedTask
from agentlab2.tools.bgym_core.utils import flatten_axtree_to_str, flatten_dom_to_str

logger = logging.getLogger(__name__)


class BrowsergymConfig(ToolConfig):
    """Configuration for BrowserGym tool."""

    # Task configuration
    task_entrypoint: Callable[[], Any] | None = None
    task_kwargs: dict = {}

    # Browser configuration
    headless: bool = True
    viewport: dict | None = None
    slow_mo: int | None = None
    timeout: int | None = None
    locale: str | None = None
    timezone_id: str | None = None

    # Playwright customization
    pw_chromium_kwargs: dict = {}
    pw_context_kwargs: dict = {}

    # Recording
    record_video_dir: str | None = None

    # Behavioral options
    tags_to_mark: str = "standard_html"  # "all" or "standard_html"
    wait_for_user_message: bool = False
    terminate_on_infeasible: bool = True
    resizeable_window: bool = False
    action_mapping: Callable | None = None
    use_raw_page_output: bool = False
    pre_observation_delay: float = 0.0

    # Observation configuration
    use_html: bool = True
    use_axtree: bool = True
    use_screenshot: bool = True
    prune_html: bool = True
    max_wait: int = 60

    # Action configuration - which BrowserGym action subsets to use
    action_subsets: list[str] = ["bid", "nav"]

    def make(self) -> "BrowsergymTool":
        return BrowsergymTool(self)


class BrowsergymTool(AbstractTool):
    """
    BrowserGym tool wrapper that adapts BrowserGym's BrowserEnv to the AgentLab2 Tool interface.

    This tool wraps the BrowserGym environment and provides:
    - Action execution via BrowserGym's HighLevelActionSet (native actions like click, fill, hover)
    - Observation extraction (HTML, accessibility tree, screenshots)
    - Proper lifecycle management (reset, close)

    The action space is defined by BrowserGym's HighLevelActionSet.
    """

    def __init__(self, config: BrowsergymConfig) -> None:
        self.config = config
        self._env: BrowserEnv | None = None
        self._last_obs: dict | None = None
        self._last_info: dict | None = None
        # HighLevelActionSet defines the available actions and converts to Python code
        self._highlevel_action_set = HighLevelActionSet(
            subsets=config.action_subsets,
            multiaction=False,
        )
        self._action_schemas: list[ActionSchema] | None = None

    @property
    def action_set(self) -> list[ActionSchema]:
        """Generate action schemas from HighLevelActionSet."""
        if self._action_schemas is None:
            tool_descriptions = self._highlevel_action_set.to_tool_description(api="openai")
            self._action_schemas = [
                ActionSchema(
                    name=tool["name"],
                    description=tool["description"],
                    parameters=tool["parameters"],
                )
                for tool in tool_descriptions
            ]
        return self._action_schemas

    def _create_env(self) -> BrowserEnv:
        """Create a new BrowserGym environment instance."""
        # Use OpenEndedTask as default - it provides a browser without task-specific logic,
        # allowing AgentLab2's Task abstraction (e.g., MiniWobTask) to handle task setup
        task_entrypoint = self.config.task_entrypoint or OpenEndedTask
        task_kwargs = self.config.task_kwargs or {"start_url": "about:blank", "goal": None}

        env_kwargs = {
            "task_entrypoint": task_entrypoint,
            "task_kwargs": task_kwargs,
            "headless": self.config.headless,
            "tags_to_mark": self.config.tags_to_mark,
            "wait_for_user_message": self.config.wait_for_user_message,
            "terminate_on_infeasible": self.config.terminate_on_infeasible,
            "resizeable_window": self.config.resizeable_window,
            "use_raw_page_output": self.config.use_raw_page_output,
            "pre_observation_delay": self.config.pre_observation_delay,
        }
        if self.config.viewport is not None:
            env_kwargs["viewport"] = self.config.viewport
        if self.config.slow_mo is not None:
            env_kwargs["slow_mo"] = self.config.slow_mo
        if self.config.timeout is not None:
            env_kwargs["timeout"] = self.config.timeout
        if self.config.locale is not None:
            env_kwargs["locale"] = self.config.locale
        if self.config.timezone_id is not None:
            env_kwargs["timezone_id"] = self.config.timezone_id
        if self.config.pw_chromium_kwargs:
            env_kwargs["pw_chromium_kwargs"] = self.config.pw_chromium_kwargs
        if self.config.pw_context_kwargs:
            env_kwargs["pw_context_kwargs"] = self.config.pw_context_kwargs
        if self.config.record_video_dir is not None:
            env_kwargs["record_video_dir"] = self.config.record_video_dir
        if self.config.action_mapping is not None:
            env_kwargs["action_mapping"] = self.config.action_mapping

        return BrowserEnv(**env_kwargs)

    def _ensure_env(self) -> None:
        """Ensure the environment is created and reset."""
        if self._env is None:
            raise RuntimeError("BrowserGym environment is not initialized. Call reset() first.")

    @property
    def env(self) -> BrowserEnv:
        """Access the underlying BrowserGym environment."""
        self._ensure_env()
        return self._env  # type: ignore

    @property
    def page(self) -> Any:
        """Access the current Playwright page from BrowserGym."""
        self._ensure_env()
        return self._env.page if self._env and hasattr(self._env, "page") else None

    def reset(self) -> None:
        """Reset the environment."""
        if self._env is not None:
            self._env.close()
        self._env = self._create_env()
        self._last_obs, self._last_info = self._env.reset()

    def execute_action(self, action: Action) -> Observation:
        """Execute an action and return the observation."""
        self._ensure_env()

        # Convert Action to BrowserGym action string
        action_str = self._action_to_string(action)

        try:
            # Use HighLevelActionSet to convert to executable Python code
            python_code = self._highlevel_action_set.to_python_code(action_str)
            obs, reward, terminated, truncated, info = self._env.step(python_code)  # type: ignore

            # Store last observation for utility methods
            self._last_obs = obs
            self._last_info = info

            # Build observation with action result
            action_result = f"Action executed: {action_str}"
            if terminated:
                action_result += " (Episode terminated)"
            if truncated:
                action_result += " (Episode truncated)"

            obs_result = Observation(contents=[Content(data=action_result, tool_call_id=action.id)])
            # Convert BrowserGym observation dict to AgentLab2 Observation
            obs_result += self._bgym_obs_to_agentlab_obs(obs)
            return obs_result

        except Exception as e:
            error_msg = f"Error executing action {action.name}: {e}"
            logger.exception(error_msg)
            return Observation(contents=[Content(data=error_msg, tool_call_id=action.id)])

    def _action_to_string(self, action: Action) -> str:
        """Convert an Action to a BrowserGym action string like 'click("a51")'."""
        args_parts = []
        for key, value in action.arguments.items():
            if isinstance(value, str):
                args_parts.append(f'{key}="{value}"')
            else:
                args_parts.append(f"{key}={repr(value)}")
        args_str = ", ".join(args_parts)
        return f"{action.name}({args_str})"

    def _bgym_obs_to_agentlab_obs(self, bgym_obs: dict) -> Observation:
        """
        Convert BrowserGym observation dict to AgentLab2 Observation.

        BrowserGym provides observations with keys like:
        - 'screenshot': numpy array (converted to PIL Image)
        - 'dom_object': dict with 'documents' and 'strings' (converted to HTML string)
        - 'axtree_object': dict (converted to accessibility tree string)
        """
        obs = Observation()

        # Add HTML if configured (flatten BrowserGym's DOM to HTML string)
        if self.config.use_html and "dom_object" in bgym_obs:
            dom_obj = bgym_obs["dom_object"]
            # Use flatten_dom_to_str to convert BrowserGym's compact DOM format to HTML
            html_str = flatten_dom_to_str(
                dom_obj,
                extra_properties=bgym_obs.get("extra_element_properties", {}),
                with_visible=False,
                filter_visible_only=False,
            )
            # BrowserGym's flatten_dom_to_str already provides a pruned representation
            obs.contents.append(Content(data=html_str, name="pruned_html"))

        # Add accessibility tree if configured
        if self.config.use_axtree and "axtree_object" in bgym_obs:
            axtree_obj = bgym_obs["axtree_object"]
            if axtree_obj:
                axtree_str = flatten_axtree_to_str(axtree_obj)
                obs.contents.append(Content(data=axtree_str, name="axtree_txt"))

        # Add screenshot if configured (convert numpy array to PIL Image)
        if self.config.use_screenshot and "screenshot" in bgym_obs:
            screenshot = bgym_obs["screenshot"]
            if isinstance(screenshot, Image.Image):
                obs.contents.append(Content(data=screenshot, name="screenshot"))
            elif isinstance(screenshot, np.ndarray):
                screenshot_img = Image.fromarray(screenshot)
                obs.contents.append(Content(data=screenshot_img, name="screenshot"))

        return obs

    # === Utility methods for BrowserTaskTool (goto, evaluate_js, page_obs) ===

    def goto(self, url: str) -> None:
        """Navigate to a URL and wait for the page to be fully loaded."""
        self._ensure_env()
        self.page.goto(url)

    def evaluate_js(self, js: str) -> Any:
        """Evaluate JavaScript in the browser context and return the result."""
        self._ensure_env()
        if self.page is not None:
            return self.page.evaluate(js)
        return None

    def page_obs(self) -> Observation:
        """
        Get the current page observation from BrowserGym.

        This method acts as an adapter, converting BrowserGym's observation format
        to AgentLab2's Observation format.
        """
        self._ensure_env()

        # Get the latest observation from BrowserGym
        # If we have a cached observation from the last step, use it
        # Otherwise, the observation should be available from reset()
        if self._last_obs is None:
            raise RuntimeError("No BrowserGym observation available. Did you call reset()?")

        return self._bgym_obs_to_agentlab_obs(self._last_obs)

    def close(self) -> None:
        """Clean up BrowserGym environment resources."""
        if self._env is not None:
            try:
                self._env.close()
            except Exception as e:
                logger.warning(f"Error closing BrowserGym environment: {e}")
            finally:
                self._env = None
                self._last_obs = None
                self._last_info = None
