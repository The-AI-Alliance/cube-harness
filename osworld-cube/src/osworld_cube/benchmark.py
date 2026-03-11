"""OSWorld benchmark for desktop/VM-based agent evaluation.

Ported from agentlab2.benchmarks.osworld.benchmark. Desktop_env dependency removed;
the tool is provided via OSWorldComputerConfig (cube-computer-tool backend).
"""

import json
import logging
import os
import random
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Optional

from pydantic import Field

from agentlab2.benchmark import Benchmark
from osworld_cube.config import OSWorldComputerConfig
from osworld_cube.task import OSWorldTask
from osworld_cube.vm_utils import OSWORLD_COMMIT, OSWORLD_REPO_DIR

logger = logging.getLogger(__name__)

OSWORLD_SYSTEM_PROMPT_COMPUTER_13 = """You are a desktop automation agent controlling a real Ubuntu computer.

## Observations
Each step you receive:
- A screenshot of the current screen (with numbered element markers when Set-of-Marks is active)
- An element table listing interactive UI elements with their index, tag, name, text, and position

## Coordinate System
- Screen resolution is typically 1920x1080
- Element positions in the table are given as top-left (x, y) and size (w, h)
- Click the center of an element: click(x=x + w//2, y=y + h//2)
- When Set-of-Marks is active: numbered boxes on the screenshot correspond to row indices in the element table

## Actions
- click(x, y): Left-click at pixel coordinates
- right_click(x, y): Right-click at pixel coordinates
- double_click(x, y): Double-click at pixel coordinates
- typing(text): Type text (the target element must already be focused)
- hotkey(keys): Press keys simultaneously, e.g., ["ctrl", "c"]
- press(key): Press a single key, e.g., "enter", "esc", "tab", "backspace"
- scroll(dx, dy): Scroll the mouse wheel (positive dy = scroll down)
- drag_to(x, y): Click-and-drag from current cursor position to (x, y)
- fail(): Signal this task CANNOT be completed — call this for impossible/infeasible tasks
- done(): Signal the task is successfully COMPLETE and stop

## Strategy
1. Study the screenshot carefully before deciding on an action
2. Use the element table to find precise coordinates — prefer clicking element centers
3. If the task is clearly impossible, call fail() immediately without wasting turns
4. Prefer keyboard shortcuts over mouse clicks when practical (faster, more reliable)
5. After completing the task, verify success in the screenshot then call done()
6. Do not loop — if an action has no effect after 2 attempts, try a different approach"""

# Keep old name as alias
OSWORLD_SYSTEM_PROMPT = OSWORLD_SYSTEM_PROMPT_COMPUTER_13


class OSWorldBenchmark(Benchmark):
    """OSWorld benchmark for desktop/VM-based agent evaluation.

    Reference: https://github.com/xlang-ai/OSWorld

    Attributes:
        tool_config:    Configuration for the Computer tool (VM backend + observations).
        tasks_file:     Path to a flat JSON array of tasks (overrides OSWorld repo format).
        test_set_path:  Path to OSWorld evaluation_examples directory.
        test_set_name:  Name of test set file (default: ``"test_all.json"``).
        domain:         Task domain filter (``"all"`` or e.g. ``"chrome"``, ``"os"``).
        shuffle:        Shuffle tasks before returning.
        shuffle_seed:   Seed for reproducible shuffling.
        max_turns:      Maximum agent turns per task.
        use_som:        Enable Set-of-Marks screenshot annotation.
    """

    tool_config: OSWorldComputerConfig = Field(default_factory=OSWorldComputerConfig)

    tasks_file: Optional[str] = None
    test_set_path: str = str(OSWORLD_REPO_DIR / "evaluation_examples")
    test_set_name: str = "test_all.json"
    domain: str = "all"
    shuffle: bool = True
    shuffle_seed: int = 42
    max_turns: int = 15
    use_som: bool = False

    def setup(self) -> None:
        """Initialize benchmark — clone OSWorld repo if needed."""
        logger.info("Setting up OSWorldBenchmark (domain=%s)", self.domain)

        if self.tasks_file:
            if not Path(self.tasks_file).exists():
                raise FileNotFoundError(f"Tasks file not found: {self.tasks_file}")
            return

        if not OSWORLD_REPO_DIR.exists():
            logger.info("OSWorld repo not found, cloning to %s…", OSWORLD_REPO_DIR)
            self._ensure_osworld_installed()

        test_set_full_path = Path(self.test_set_path) / self.test_set_name
        if not test_set_full_path.exists():
            raise FileNotFoundError(
                f"Test set not found: {test_set_full_path}\n"
                f"Try deleting {OSWORLD_REPO_DIR} and running again."
            )

    def close(self) -> None:
        logger.info("Closing OSWorldBenchmark")

    def load_tasks(self) -> list[OSWorldTask]:
        if self.tasks_file:
            return self._load_tasks_from_file(self.tasks_file)
        return self._load_tasks_from_repo()

    def _load_tasks_from_file(self, tasks_file: str) -> list[OSWorldTask]:
        with open(tasks_file) as f:
            task_list = json.load(f)

        tasks = []
        for task_data in task_list:
            task_domain = task_data.get("domain", "general")
            if self.domain != "all" and task_domain != self.domain:
                continue
            tasks.append(OSWorldTask(
                id=task_data["id"],
                desc=task_data.get("desc", task_data.get("instruction", "")),
                domain=task_domain,
                instruction=task_data.get("instruction", ""),
                snapshot=task_data.get("snapshot", "init_state"),
                related_apps=task_data.get("related_apps", []),
                config=task_data.get("config", []),
                evaluator=task_data.get("evaluator", {}),
                max_turns=self.max_turns,
                use_som=self.use_som,
            ))

        if self.shuffle:
            random.seed(self.shuffle_seed)
            random.shuffle(tasks)

        logger.info("Loaded %d tasks from %s", len(tasks), tasks_file)
        return tasks

    def _load_tasks_from_repo(self) -> list[OSWorldTask]:
        test_set_file = Path(self.test_set_path) / self.test_set_name

        if not test_set_file.exists():
            logger.error("Test set file not found: %s", test_set_file)
            return []

        with open(test_set_file) as f:
            tasks_by_domain = json.load(f)

        if self.domain != "all":
            if self.domain in tasks_by_domain:
                tasks_by_domain = {self.domain: tasks_by_domain[self.domain]}
            else:
                logger.warning("Domain '%s' not found in test set", self.domain)
                return []

        tasks = []
        for domain_name, task_ids in tasks_by_domain.items():
            logger.info("Loading %d tasks from domain '%s'", len(task_ids), domain_name)
            for task_id in task_ids:
                task_file = Path(self.test_set_path) / f"examples/{domain_name}/{task_id}.json"
                if not task_file.exists():
                    logger.warning("Task file not found: %s", task_file)
                    continue
                try:
                    with open(task_file) as f:
                        task_data = json.load(f)
                    task_data = self._fix_settings_paths(task_data)
                    tasks.append(OSWorldTask(
                        id=task_data.get("id", task_id),
                        desc=task_data.get("instruction", ""),
                        domain=domain_name,
                        instruction=task_data.get("instruction", ""),
                        snapshot=task_data.get("snapshot", "init_state"),
                        related_apps=task_data.get("related_apps", []),
                        config=task_data.get("config", []),
                        evaluator=task_data.get("evaluator", {}),
                        max_turns=self.max_turns,
                        use_som=self.use_som,
                    ))
                except Exception as exc:
                    logger.error("Failed to load task %s: %s", task_id, exc)

        if self.shuffle:
            random.seed(self.shuffle_seed)
            random.shuffle(tasks)

        logger.info("Loaded %d total tasks (domain=%s)", len(tasks), self.domain)
        return tasks

    def _fix_settings_paths(self, task: dict) -> dict:
        """Prepend OSWorld repo path to relative settings_file paths in task config."""
        updated_task = deepcopy(task)
        for config_item in updated_task.get("config", []):
            parameters = config_item.get("parameters", {})
            if "settings_file" in parameters:
                settings_file = parameters["settings_file"]
                repo_dir = Path(os.environ.get("OSWORLD_REPO", str(OSWORLD_REPO_DIR)))
                parameters["settings_file"] = str(repo_dir / settings_file)
        return updated_task

    def _ensure_osworld_installed(self) -> None:
        """Clone OSWorld repo and pin to the known-good commit."""
        OSWORLD_REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "https://github.com/xlang-ai/OSWorld", str(OSWORLD_REPO_DIR)],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(
            ["git", "checkout", OSWORLD_COMMIT],
            cwd=str(OSWORLD_REPO_DIR), check=True, capture_output=True, text=True,
        )
        logger.info("OSWorld repository cloned and pinned to commit %s", OSWORLD_COMMIT)
