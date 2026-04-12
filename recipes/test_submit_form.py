"""Quick test script for submit_form() on create-incident task.

Calls submit_form() on a fresh form (no fields filled) to check whether
WorkArena's gsftSubmit patch writes localStorage before old_gsftSubmit throws.
Look for INFO logs from cube_harness.tools.browsergym showing:
  - "gsftSubmit threw (expected): ..."
  - "localStorage sys_id keys after call: ..."

Usage:
    uv run recipes/test_submit_form.py
"""

import logging
import time

from cube.tool import ToolboxConfig
from cube_browser_playwright.playwright_session import PlaywrightSessionConfig
from workarena_cube.benchmark import WorkArenaBenchmark

from cube_harness.tools.browsergym import BrowsergymConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")

tool_config = ToolboxConfig(
    tool_configs=[
        BrowsergymConfig(
            browser=PlaywrightSessionConfig(headless=False, timeout=30000),
            use_screenshot=False,
            use_axtree=True,
            use_html=False,
            pre_observation_delay=1.0,
        ),
    ]
)

b = WorkArenaBenchmark(default_tool_config=tool_config, level="l1", n_seeds_l1=1)
b.setup()

configs = b.get_task_configs()
task_config = next(c for c in configs if c.task_id == "workarena.servicenow.create-incident")

print(f"Task: {task_config.task_id}")
print("=" * 60)

task = task_config.make()
obs, info = task.reset()
print(f"Task reset. Goal: {info.get('goal', '')[:120]}")

# Get the tool from the task
from cube_harness.tools.browsergym import BrowsergymTool
from cube.tool import Toolbox

tool = task.tool
if isinstance(tool, Toolbox):
    bgym_tool = tool.find_tool(BrowsergymTool)
else:
    bgym_tool = tool

print(f"\nTool type: {type(bgym_tool).__name__}")
print(f"Frames on page: {[f.name for f in bgym_tool.page.frames]}")

# Wait for page to settle
time.sleep(2)

print("\nCalling submit_form()...")
result = bgym_tool.submit_form()
print(f"Result: {result}")

reward, eval_info = task.evaluate(obs)
print(f"\nReward: {reward}")
print(f"Done: {eval_info.get('done')}")
print(f"Message: {eval_info.get('message', '')}")

task.close()
b.close()
