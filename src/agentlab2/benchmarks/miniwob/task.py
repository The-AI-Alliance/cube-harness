import logging

from PIL import Image

from agentlab2.action_spaces.browser_action_space import BrowserActionSpace
from agentlab2.core import Action, ActionSchema, ActionSubset, Content, Observation, Task
from agentlab2.tools.playwright import SyncPlaywrightTool

logger = logging.getLogger(__name__)


class MiniWobTask(Task):
    validate_per_step: bool = True
    supported_actions: ActionSubset = (
        BrowserActionSpace.browser_press_key,
        BrowserActionSpace.browser_type,
        BrowserActionSpace.browser_click,
        BrowserActionSpace.browser_drag,
        BrowserActionSpace.browser_hover,
        BrowserActionSpace.browser_select_option,
        BrowserActionSpace.browser_mouse_click_xy,
    )
    _tool: SyncPlaywrightTool

    def __init__(
        self,
        id: str,
        desc: str,
        subdomain: str,
        base_url: str,
        remove_human_display: bool = True,
        episode_max_time: int = 1000000,
        max_turns: int = 10,
    ) -> None:
        self.id = id
        self.desc = desc
        self.subdomain = subdomain
        self.base_url = base_url[:-1] if base_url.endswith("/") else base_url
        self.remove_human_display = remove_human_display
        self.episode_max_time = episode_max_time
        self.max_turns = max_turns

    @property
    def url(self) -> str:
        return f"{self.base_url}/{self.subdomain}.html"

    def setup(self, tool: SyncPlaywrightTool) -> tuple[Observation, dict]:  # This needs
        """
        Set up everything needed to execute the task.

        Args:
            page: the active playwright page.

        Returns:
            goal: str, goal of the task.
            info: dict, custom information from the task.
        """
        self._tool = tool
        logger.info(f"Setting up MiniWob task {self.id} at {self.url}")
        self._tool.goto(self.url)
        setup_js = self._get_setup_js()
        setup_result = self._tool.evaluate_js(setup_js)
        goal, info = self._parse_setup_result(setup_result)
        obs = Observation.from_text(goal)
        obs += self.obs_postprocess(self._tool.page_obs())
        return obs, {**info, "task_id": self.id, "task_url": self.url, "task_desc": self.desc}

    def validate_task(self, obs: Observation, actions: list[Action] | None = None) -> tuple[float, dict]:
        """
        Validate the task, either per step or at the end.

        Returns:
            reward: float, the reward obtained.
            info: dict, custom information from the validation.
        """
        validate_result = self._tool.evaluate_js("""() => {
return [WOB_REWARD_GLOBAL, WOB_RAW_REWARD_GLOBAL, WOB_REWARD_REASON, WOB_DONE_GLOBAL, WOB_EPISODE_ID, WOB_TASK_READY];
}""")
        reward, info = self._parse_validation_result(validate_result)
        return reward, info

    def _get_setup_js(self) -> str:
        if self.remove_human_display:
            js = r"""
let __display_ids = ['reward-display', 'click-canvas', 'sync-task-cover'];
let __display_divs = {};
let __query_div_hidden_copy = null;

removeDisplay = function() {
  core.clearTimer();
  document.body.removeEventListener('click', core.canvasDrawClick);

  __query_div_hidden_copy = document.getElementById('query').cloneNode(true);
  document.getElementById('query').innerHTML = '';

  for (i in __display_ids) {
    elem_id = __display_ids[i];
    elem = document.getElementById(elem_id);
    // remove elem from the document
    elem.remove();
    // but keep it stored somewhere to bring back later
    __display_divs[elem_id] = elem;
  }
};

bringBackDisplay = function() {
  document.getElementById('query').innerHTML = __query_div_hidden_copy.innerHTML;
  for (var elem_id in __display_divs){
    document.body.appendChild(__display_divs[elem_id]);
  }
  core.createDisplay();
};

core.endEpisode_legacy = core.endEpisode;
core.startEpisodeReal_legacy = core.startEpisodeReal;
core.getUtterance_legacy = core.getUtterance;

core.getUtterance = function () {
  bringBackDisplay();
  utterance = core.getUtterance_legacy();
  removeDisplay();
  return utterance;
};

core.endEpisode = function(reward, time_proportional, reason){
  bringBackDisplay();
  core.endEpisode_legacy(reward, time_proportional, reason);
  removeDisplay();
};

core.startEpisodeReal = function() {
  bringBackDisplay();
  core.startEpisodeReal_legacy();
  removeDisplay();
};

removeDisplay();
"""
        else:
            js = ""
        js += f"""
Math.seedrandom(42);
core.EPISODE_MAX_TIME = {self.episode_max_time};
core.startEpisodeReal();
while (!WOB_TASK_READY) {{
  await new Promise(resolve => setTimeout(resolve, 100));
}}
return core.getUtterance();
    """
        return f"async () => {{{js}}}"

    def _parse_setup_result(self, setup_result: str | dict | list) -> tuple[str, dict]:
        if isinstance(setup_result, dict):
            return setup_result["utterance"], {}
        elif isinstance(setup_result, str):
            return setup_result, {}
        else:
            raise ValueError(f"Unexpected setup_result type: {type(setup_result)}")

    def _parse_validation_result(self, validation_result: str | dict | list) -> tuple[float, dict]:
        if isinstance(validation_result, list):
            chunks = validation_result
            done = chunks[3]
        elif isinstance(validation_result, dict):
            raise ValueError("Validation result as dict is not supported")
        else:
            chunks = [c.strip() for c in validation_result.split(",")]
            done = chunks[3].strip().lower() == "true"
        raw_reward = float(chunks[1])
        reward = float(raw_reward > 0)
        return reward, {
            "raw_reward": raw_reward,
            "reward_reason": chunks[2],
            "done": done,
        }

    def obs_postprocess(self, obs: Observation) -> Observation:
        contents = []
        for content in obs.contents:
            if content.name == "screenshot" and isinstance(content.data, Image.Image):
                # crop to 332x214 because this is the viewport size for MiniWob
                cropped_image = content.data.crop((0, 0, 332, 214))
                contents.append(Content(name=content.name, data=cropped_image))
            else:
                contents.append(content)
        obs.contents = contents
        return obs

    def filter_actions(self, actions: list[ActionSchema]) -> list[ActionSchema]:
        supported_action_names = {action.__name__ for action in self.supported_actions}
        filtered = [a for a in actions if a.name in supported_action_names]
        logger.info(f"Chosen {len(filtered)} out of {len(actions)} actions for MiniWob task.")
        return filtered

    def finished(self) -> bool:
        return self._tool.evaluate_js("() => {return WOB_DONE_GLOBAL;}")
