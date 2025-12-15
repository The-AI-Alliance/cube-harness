from agentlab2.core import Action, EnvironmentOutput
from agentlab2.environment import EnvironmentConfig, Task, ToolboxEnv
from agentlab2.tools.playwright import PWConfig, SyncPlaywrightTool


class BrowserEnvConfig(EnvironmentConfig):
    """Configuration for BrowserEnv."""

    pw_config: PWConfig

    def make(self) -> "BrowserEnv":
        """Create a BrowserEnv instance from the configuration for specified task."""
        browser_tool = self.pw_config.make()
        assert self._task is not None, "Task must be set in EnvironmentConfig before making the environment."
        return BrowserEnv(self._task, browser_tool)


class BrowserEnv(ToolboxEnv):
    """Environment that uses only one browser tool for interaction."""

    def __init__(self, task: Task, browser_tool: SyncPlaywrightTool):
        super().__init__(task, [browser_tool])
        self.browser_tool = browser_tool

    def setup(self) -> EnvironmentOutput:
        output = super().setup()
        page_obs = self.browser_tool.page_obs()
        output.obs.contents += self.task.obs_postprocess(page_obs).contents
        return output

    def step(self, action: Action) -> EnvironmentOutput:
        """Action result of the browser tool does not contain page data, so:
        - append page observation after the step
        - postprocess page observation via task's obs_postprocess method
        """
        output = super().step(action)
        page_obs = self.task.obs_postprocess(self.browser_tool.page_obs())
        output.obs.contents += page_obs.contents
        return output

    def goto(self, url: str):
        self.browser_tool.goto(url)

    def evaluate_js(self, script: str):
        return self.browser_tool.evaluate_js(script)
