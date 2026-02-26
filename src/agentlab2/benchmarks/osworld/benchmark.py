"""OSWorld benchmark for desktop/VM-based agent evaluation.

This module provides the OSWorldBenchmark class that loads and manages
OSWorld tasks for desktop automation evaluation.
"""

import json
import logging
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field

from agentlab2.benchmark import Benchmark
from agentlab2.benchmarks.osworld.task import OSWorldTask
from agentlab2.tools.computer import ComputerConfig

logger = logging.getLogger(__name__)

# Provider types supported by OSWorld
ProviderType = Literal["vmware", "virtualbox", "docker", "aws", "azure", "gcp", "aliyun", "volcengine"]

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

OSWORLD_SYSTEM_PROMPT_PYAUTOGUI = """You are a desktop automation agent controlling a real Ubuntu computer.

## Observations
Each step you receive:
- A screenshot of the current screen with numbered red bounding boxes (Set-of-Marks)
- An element table listing interactive UI elements: index, tag, name, text

## Actions
You control the computer by calling run_pyautogui(code) with valid Python/pyautogui code.

### Referencing screen elements
Each numbered box in the screenshot corresponds to a tag_N variable pre-defined in your code:
- tag_1, tag_2, ... are (center_x, center_y) tuples for each element
- pyautogui.click(*tag_2) — click element #2
- pyautogui.moveTo(*tag_1) — move to element #1

### Common pyautogui commands
- pyautogui.click(*tag_N)                     — left-click element N
- pyautogui.rightClick(*tag_N)                — right-click element N
- pyautogui.doubleClick(*tag_N)               — double-click element N
- pyautogui.click(x, y)                       — click at absolute coordinates
- pyautogui.typewrite('text', interval=0.05)  — type text character by character
- pyautogui.hotkey('ctrl', 'c')               — press key combination
- pyautogui.press('enter')                    — press a single key
- pyautogui.scroll(x, y, clicks=-3)           — scroll (negative clicks = down)
- pyautogui.dragTo(x, y, button='left')       — drag to coordinates

### Ending the task
- Call fail() if the task CANNOT be completed (infeasible tasks)
- Call done() when the task is successfully COMPLETE

## Strategy
1. Study the screenshot and element table carefully before acting
2. Prefer tag_N references over raw coordinates when elements are visible
3. If the task is clearly impossible, call fail() immediately
4. Prefer hotkey shortcuts over mouse clicks when practical
5. After completing the task, verify success in the screenshot then call done()
6. Do not loop — if an action has no effect after 2 attempts, try a different approach"""

# Keep old name as alias for backwards compatibility with any existing code
OSWORLD_SYSTEM_PROMPT = OSWORLD_SYSTEM_PROMPT_COMPUTER_13

# Pinned OSWorld commit for reproducibility
OSWORLD_COMMIT = "e695a10"

# AgentLab2 data directory for OSWorld
AGENTLAB2_DIR = Path.home() / ".agentlab2"
OSWORLD_BASE_DIR = AGENTLAB2_DIR / "benchmarks" / "osworld"
OSWORLD_REPO_DIR = OSWORLD_BASE_DIR / "OSWorld"
OSWORLD_VM_DIR = OSWORLD_BASE_DIR / "vm_data"
OSWORLD_CACHE_DIR = OSWORLD_BASE_DIR / "cache"


class OSWorldBenchmark(Benchmark):
    """OSWorld benchmark for desktop/VM-based agent evaluation.

    OSWorld tasks run inside Docker containers or VMs, evaluating agents
    on real-world desktop tasks across different operating systems.

    Reference: https://github.com/xlang-ai/OSWorld

    Attributes:
        tool_config: Configuration for the Computer tool
        tasks_file: Path to a flat JSON array of tasks (overrides test_set_path/test_set_name)
        test_set_path: Path to OSWorld task definitions (used when tasks_file is not set)
        test_set_name: Name of test set file (e.g., "test_all.json")
        domain: Task domain filter ("all", "chrome", "os", "libreoffice", etc.)
        shuffle: Whether to shuffle tasks before returning them
        shuffle_seed: Seed for reproducible shuffling
    """

    # Tool configuration (required by Benchmark base class)
    tool_config: ComputerConfig = Field(default_factory=ComputerConfig)

    # Task selection - either a direct tasks file or OSWorld repo format
    tasks_file: Optional[str] = None  # Path to flat JSON array of tasks
    test_set_path: str = str(OSWORLD_REPO_DIR / "evaluation_examples")
    test_set_name: str = "test_all.json"
    domain: str = "all"  # or specific domain like "chrome", "libreoffice", etc.
    shuffle: bool = True
    shuffle_seed: int = 42
    max_turns: int = 15  # Maximum agent turns per task
    use_som: bool = False  # Enable Set-of-Marks screenshot annotation

    def setup(self) -> None:
        """Initialize benchmark.

        If tasks_file is set, validates the file exists.
        Otherwise, automatically clones OSWorld repo if not present.
        Desktop_env VMs are managed per-task by the Computer tool.
        """
        logger.info(
            f"Setting up OSWorld benchmark with provider={self.tool_config.provider}, "
            f"domain={self.domain}"
        )

        if self.tasks_file:
            # Using a custom tasks file - just validate it exists
            if not Path(self.tasks_file).exists():
                raise FileNotFoundError(f"Tasks file not found: {self.tasks_file}")
            return

        # Ensure OSWorld repo exists (auto-install if needed)
        if not OSWORLD_REPO_DIR.exists():
            logger.info(f"OSWorld repo not found, cloning to {OSWORLD_REPO_DIR}...")
            self._ensure_osworld_installed()

        # Verify task files exist
        test_set_full_path = Path(self.test_set_path) / self.test_set_name
        if not test_set_full_path.exists():
            raise FileNotFoundError(
                f"Test set not found: {test_set_full_path}\n"
                f"OSWorld repo exists but task files are missing. "
                f"Try deleting {OSWORLD_REPO_DIR} and running again."
            )

    def close(self) -> None:
        """Cleanup benchmark resources.

        VMs are closed per-task by Computer tool's close() method.
        """
        logger.info("Closing OSWorld benchmark")
        # VMs are closed per-task, no global cleanup needed

    def load_tasks(self) -> list[OSWorldTask]:
        """Load and return all OSWorld tasks.

        If tasks_file is set, loads tasks from a flat JSON array.
        Otherwise, loads from OSWorld repo's directory structure.

        Returns:
            List of OSWorldTask instances
        """
        if self.tasks_file:
            return self._load_tasks_from_file(self.tasks_file)
        return self._load_tasks_from_repo()

    def _load_tasks_from_file(self, tasks_file: str) -> list[OSWorldTask]:
        """Load tasks from a flat JSON array file.

        Expected format: [{"id": "...", "instruction": "...", ...}, ...]

        Args:
            tasks_file: Path to JSON file with task array

        Returns:
            List of OSWorldTask instances
        """
        with open(tasks_file) as f:
            task_list = json.load(f)

        tasks = []
        for task_data in task_list:
            # Filter by domain if not "all"
            task_domain = task_data.get("domain", "general")
            if self.domain != "all" and task_domain != self.domain:
                continue

            task = OSWorldTask(
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
            )
            tasks.append(task)

        if self.shuffle:
            random.seed(self.shuffle_seed)
            random.shuffle(tasks)

        logger.info(f"Loaded {len(tasks)} tasks from {tasks_file}")
        return tasks

    def _load_tasks_from_repo(self) -> list[OSWorldTask]:
        """Load tasks from OSWorld repo's directory structure.

        Loads from test_set_path using the domain→task ID mapping format.

        Returns:
            List of OSWorldTask instances
        """
        test_set_file = Path(self.test_set_path) / self.test_set_name

        if not test_set_file.exists():
            logger.error(f"Test set file not found: {test_set_file}")
            logger.error(
                "Please ensure OSWorld repository is cloned and test files are available"
            )
            return []

        # Load test set (maps domains to task IDs)
        with open(test_set_file) as f:
            tasks_by_domain = json.load(f)

        # Filter by domain if not "all"
        if self.domain != "all":
            if self.domain in tasks_by_domain:
                tasks_by_domain = {self.domain: tasks_by_domain[self.domain]}
            else:
                logger.warning(f"Domain '{self.domain}' not found in test set")
                return []

        tasks = []
        for domain_name in tasks_by_domain:
            task_ids = tasks_by_domain[domain_name]
            logger.info(f"Loading {len(task_ids)} tasks from domain '{domain_name}'")

            for task_id in task_ids:
                # Load individual task file
                task_file = Path(self.test_set_path) / f"examples/{domain_name}/{task_id}.json"

                if not task_file.exists():
                    logger.warning(f"Task file not found: {task_file}")
                    continue

                try:
                    with open(task_file) as f:
                        task_data = json.load(f)

                    # Fix file paths in config (prepend OSWORLD_REPO)
                    task_data = self._fix_settings_paths(task_data)

                    # Create OSWorldTask
                    task = OSWorldTask(
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
                    )
                    tasks.append(task)

                except Exception as e:
                    logger.error(f"Failed to load task {task_id}: {e}")
                    continue

        if self.shuffle:
            random.seed(self.shuffle_seed)
            random.shuffle(tasks)

        logger.info(f"Loaded {len(tasks)} total tasks from domain '{self.domain}'")
        return tasks

    def _fix_settings_paths(self, task: dict) -> dict:
        """Fix file paths in task config to use OSWorld repo directory.

        OSWorld task configs often reference files with relative paths.
        This method prepends the OSWorld repo path to these paths.

        Args:
            task: Task configuration dict

        Returns:
            Updated task dict with fixed paths
        """
        updated_task = deepcopy(task)

        for config_item in updated_task.get("config", []):
            parameters = config_item.get("parameters", {})
            if "settings_file" in parameters:
                # Prepend OSWorld repo path to settings_file
                settings_file = parameters["settings_file"]
                repo_dir = Path(os.environ.get("OSWORLD_REPO", str(OSWORLD_REPO_DIR)))
                parameters["settings_file"] = str(repo_dir / settings_file)

        return updated_task

    def _ensure_osworld_installed(self) -> None:
        """Ensure OSWorld repo is cloned and directories exist.

        This is called automatically by setup() if needed.
        """
        import subprocess

        # Create base directory
        OSWORLD_BASE_DIR.mkdir(parents=True, exist_ok=True)

        # Clone OSWorld repository if not exists
        if not OSWORLD_REPO_DIR.exists():
            logger.info(f"Cloning OSWorld repository to {OSWORLD_REPO_DIR}...")
            try:
                subprocess.run(
                    ["git", "clone", "https://github.com/xlang-ai/OSWorld", str(OSWORLD_REPO_DIR)],
                    check=True,
                    capture_output=True,
                    text=True
                )
                subprocess.run(
                    ["git", "checkout", OSWORLD_COMMIT],
                    cwd=str(OSWORLD_REPO_DIR),
                    check=True,
                    capture_output=True,
                    text=True
                )
                logger.info(f"✓ OSWorld repository cloned and pinned to commit {OSWORLD_COMMIT}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to clone OSWorld: {e.stderr}")
                raise

        # Create VM data directory (VMs will be downloaded automatically by desktop_env)
        OSWORLD_VM_DIR.mkdir(parents=True, exist_ok=True)

    def install(self) -> None:
        """Install OSWorld dependencies.

        Clones the OSWorld repository and sets up directory structure.
        VM images are downloaded automatically by desktop_env on first use.

        Note: This is called automatically by setup() if needed, so you
        typically don't need to call this explicitly.
        """
        logger.info("Installing OSWorld benchmark...")
        self._ensure_osworld_installed()
        logger.info("\nInstallation complete!")
        logger.info(f"  OSWorld repo: {OSWORLD_REPO_DIR}")
        logger.info(f"  VM data: {OSWORLD_VM_DIR}")
        logger.info("\nVM images will be downloaded automatically on first use.")
        logger.info("Make sure desktop_env is installed: pip install desktop-env")
        logger.warning(
            "\n"
            "⚠️  IMPORTANT: Google Chrome setup required for accurate results!\n"
            "   Chrome tasks will fail or produce incorrect scores without this step.\n"
            f"   Follow the Chrome configuration instructions in the OSWorld README:\n"
            f"   {OSWORLD_REPO_DIR / 'README.md'}\n"
            "   (search for 'Google Chrome' in that file)"
        )

    def uninstall(self) -> None:
        """Remove OSWorld resources.

        Removes downloaded VM images and cleans up installation.
        """
        logger.info("Uninstalling OSWorld...")
        logger.info("To remove OSWorld:")
        logger.info("1. Delete VM images")
        logger.info("2. Remove OSWorld repository")
        logger.info("3. Uninstall desktop_env: pip uninstall desktop-env")
