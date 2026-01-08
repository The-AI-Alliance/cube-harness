import logging
from typing import Any, Callable

from PIL import Image

from agentlab2.action_spaces.browser_action_space import BrowserActionSpace
from agentlab2.bgym_core.env import BrowserEnv
from agentlab2.core import Action, ActionSchema, Content, Observation
from agentlab2.tool import Tool, ToolConfig
from agentlab2.utils import prune_html

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

    def make(self) -> "BrowsergymTool":
        return BrowsergymTool(self)


class BrowsergymTool(Tool, BrowserActionSpace):
    """
    BrowserGym tool wrapper that adapts BrowserGym's BrowserEnv to the AgentLab2 Tool interface.

    This tool wraps the BrowserGym environment and provides:
    - Action execution via BrowserGym's action system
    - Observation extraction (HTML, accessibility tree, screenshots)
    - Proper lifecycle management (reset, close)

    Note: BrowserGym uses string-based actions (Python code). This tool provides
    a method to execute raw BrowserGym actions or use the higher-level action set.
    """

    action_space = BrowserActionSpace

    def __init__(self, config: BrowsergymConfig) -> None:
        self.config = config
        self._env: Any = None
        self._last_obs: dict | None = None
        self._last_info: dict | None = None
        self._action_schemas: list[ActionSchema] | None = None

    def _create_env(self) -> BrowserEnv:
        """Create a new BrowserGym environment instance."""
        env_kwargs = {
            "headless": self.config.headless,
            "tags_to_mark": self.config.tags_to_mark,
            "wait_for_user_message": self.config.wait_for_user_message,
            "terminate_on_infeasible": self.config.terminate_on_infeasible,
            "resizeable_window": self.config.resizeable_window,
            "use_raw_page_output": self.config.use_raw_page_output,
            "pre_observation_delay": self.config.pre_observation_delay,
        }

        if self.config.task_entrypoint is not None:
            env_kwargs["task_entrypoint"] = self.config.task_entrypoint
        if self.config.task_kwargs:
            env_kwargs["task_kwargs"] = self.config.task_kwargs
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
    def env(self) -> Any:
        """Access the underlying BrowserGym environment."""
        self._ensure_env()
        return self._env

    @property
    def page(self) -> Any:
        """Access the current Playwright page from BrowserGym."""
        self._ensure_env()
        # BrowserGym stores the active page in the environment
        return self._env.page if hasattr(self._env, "page") else None

    def reset(self) -> Observation:
        """Reset the environment and return initial observation."""
        if self._env is not None:
            self._env.close()
        else:
            self._env = self._create_env()
        bgym_obs, info = self._env.reset()
        return self._extract_observation(bgym_obs)

    def execute_action(self, action: Action) -> Observation:
        """
        Execute an action and return the observation.

        The action is converted to a BrowserGym action string and executed.
        """
        self._ensure_env()

        # Convert Action to BrowserGym action string
        action_str = self._action_to_browsergym(action)

        try:
            obs, reward, terminated, truncated, info = self._env.step(action_str)

            # Build observation with action result
            action_result = f"Action executed. Reward: {reward}"
            if terminated:
                action_result += " (Episode terminated)"
            if truncated:
                action_result += " (Episode truncated)"

            obs_result = Observation(contents=[Content(data=action_result, tool_call_id=action.id)])
            obs_result += self._extract_observation(obs)
            return obs_result

        except Exception as e:
            error_msg = f"Error executing action {action.name}: {e}"
            logger.exception(error_msg)
            return Observation(contents=[Content(data=error_msg, tool_call_id=action.id)])

    def _action_to_browsergym(self, action: Action) -> str:
        """
        Convert an AgentLab2 Action to a BrowserGym action string.

        BrowserGym expects Python code strings. This method converts our
        structured actions to the appropriate BrowserGym format.
        """
        action_handlers = {
            "browser_click": self.browser_click,
            "browser_type": self.browser_type,
            "browser_press_key": self.browser_press_key,
            "browser_hover": self.browser_hover,
            "browser_mouse_click_xy": self.browser_mouse_click_xy,
            "browser_drag": self.browser_drag,
            "browser_select_option": self.browser_select_option,
            "browser_wait": self.browser_wait,
            "browser_back": self.browser_back,
            "browser_forward": self.browser_forward,
            "noop": self.noop,
            "goto": self.goto,
            "evaluate_js": self.evaluate_js,
        }

        handler = action_handlers.get(action.name)
        if not handler:
            raise ValueError(f"Unsupported action name: {action.name}")
        return handler(action.arguments)

    def _extract_bid(self, selector: str) -> str:
        """Extract bid from a [bid=X] format selector."""
        return selector.split("=")[1].rstrip("]\"'")

    def browser_click(self, args: dict) -> str:
        """Convert browser_click action to BrowserGym format."""
        selector = args.get("selector", "")
        if selector.startswith("[bid="):
            bid = self._extract_bid(selector)
            return f'click("{bid}")'
        return f'click("{selector}")'

    def browser_type(self, args: dict) -> str:
        """Convert browser_type action to BrowserGym format."""
        selector = args.get("selector", "")
        text = args.get("text", "")
        if selector.startswith("[bid="):
            bid = self._extract_bid(selector)
            return f'fill("{bid}", "{text}")'
        return f'fill("{selector}", "{text}")'

    def browser_press_key(self, args: dict) -> str:
        """Convert browser_press_key action to BrowserGym format."""
        key = args.get("key", "")
        return f'press("{key}")'

    def browser_hover(self, args: dict) -> str:
        """Convert browser_hover action to BrowserGym format."""
        selector = args.get("selector", "")
        if selector.startswith("[bid="):
            bid = self._extract_bid(selector)
            return f'hover("{bid}")'
        return f'hover("{selector}")'

    def browser_mouse_click_xy(self, args: dict) -> str:
        """Convert browser_mouse_click_xy action to BrowserGym format."""
        x = args.get("x", 0)
        y = args.get("y", 0)
        return f"click({x}, {y})"

    def browser_drag(self, args: dict) -> str:
        """Convert browser_drag action to BrowserGym format."""
        from_sel = args.get("from_selector", "")
        to_sel = args.get("to_selector", "")
        return f'drag_and_drop("{from_sel}", "{to_sel}")'

    def browser_select_option(self, args: dict) -> str:
        """Convert browser_select_option action to BrowserGym format."""
        selector = args.get("selector", "")
        value = args.get("value", "")
        if selector.startswith("[bid="):
            bid = self._extract_bid(selector)
            return f'select_option("{bid}", "{value}")'
        return f'select_option("{selector}", "{value}")'

    def browser_wait(self, args: dict) -> str:
        """Convert browser_wait action to BrowserGym format."""
        seconds = args.get("seconds", 1)
        return f"noop({min(seconds, self.config.max_wait)})"

    def browser_back(self, _args: dict) -> str:
        """Convert browser_back action to BrowserGym format."""
        return "go_back()"

    def browser_forward(self, _args: dict) -> str:
        """Convert browser_forward action to BrowserGym format."""
        return "go_forward()"

    def noop(self, _args: dict) -> str:
        """Convert noop action to BrowserGym format."""
        return "noop()"

    def _extract_observation(self, browsergym_obs: dict) -> Observation:
        """Extract observation from BrowserGym observation dict."""
        obs = Observation()

        # Extract goal/task description if available
        if "goal" in browsergym_obs:
            obs.contents.append(Content(data=str(browsergym_obs["goal"]), name="goal"))

        # Extract chat messages if available
        if "chat_messages" in browsergym_obs:
            messages = browsergym_obs["chat_messages"]
            if messages:
                obs.contents.append(Content(data=str(messages), name="chat_messages"))

        # Extract URL
        if "url" in browsergym_obs:
            obs.contents.append(Content(data=str(browsergym_obs["url"]), name="url"))

        # Extract HTML/DOM
        if self.config.use_html:
            html = None
            if "dom_txt" in browsergym_obs and browsergym_obs["dom_txt"]:
                html = browsergym_obs["dom_txt"]
            elif "dom_object" in browsergym_obs and browsergym_obs["dom_object"]:
                # dom_object is typically a parsed DOM, convert to string if needed
                html = str(browsergym_obs["dom_object"])

            if html:
                if self.config.prune_html:
                    obs.contents.append(Content(data=prune_html(html), name="pruned_html"))
                else:
                    obs.contents.append(Content(data=html, name="html"))

        # Extract accessibility tree
        if self.config.use_axtree:
            if "axtree_txt" in browsergym_obs and browsergym_obs["axtree_txt"]:
                obs.contents.append(Content(data=browsergym_obs["axtree_txt"], name="axtree_txt"))
            elif "axtree_object" in browsergym_obs and browsergym_obs["axtree_object"]:
                axtree_str = _flatten_axtree(browsergym_obs["axtree_object"])
                obs.contents.append(Content(data=axtree_str, name="axtree_txt"))

        # Extract screenshot
        if self.config.use_screenshot:
            if "screenshot" in browsergym_obs and browsergym_obs["screenshot"] is not None:
                screenshot = browsergym_obs["screenshot"]
                # BrowserGym returns numpy array, convert to PIL Image
                if hasattr(screenshot, "shape"):
                    import numpy as np

                    if isinstance(screenshot, np.ndarray):
                        screenshot = Image.fromarray(screenshot)
                obs.contents.append(Content(data=screenshot, name="screenshot"))

        # Extract last action error if any
        if "last_action_error" in browsergym_obs and browsergym_obs["last_action_error"]:
            obs.contents.append(Content(data=str(browsergym_obs["last_action_error"]), name="last_action_error"))

        return obs

    def goto(self, url: str) -> None:
        """Navigate to a URL."""
        self._ensure_env()
        self._env.step(f'goto("{url}")')

    def evaluate_js(self, args: dict) -> str:
        """Convert evaluate_js action to BrowserGym format."""
        script = args.get("script", "")
        self._ensure_env()
        return self._env.step(f'evaluate_js("""{script}""")')

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


def _flatten_axtree(axtree: dict, depth: int = 0) -> str:
    # TODO: import from browsergym if available
    raise NotImplementedError
