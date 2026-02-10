"""OSWorld benchmark for desktop/VM-based agent evaluation.

This module provides the OSWorldBenchmark class that loads and manages
OSWorld tasks for desktop automation evaluation.
"""

import json
import logging
import random
from copy import deepcopy
from pathlib import Path
from typing import Literal

from pydantic import Field

from agentlab2.benchmark import Benchmark
from agentlab2.benchmarks.osworld.task import OSWorldTask
from agentlab2.tools.computer import ComputerConfig

logger = logging.getLogger(__name__)

# Provider types supported by OSWorld
ProviderType = Literal["vmware", "virtualbox", "docker", "aws"]

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
        test_set_path: Path to OSWorld task definitions
        test_set_name: Name of test set file (e.g., "test_all.json")
        domain: Task domain filter ("all", "chrome", "os", "libreoffice", etc.)
        shuffle: Whether to shuffle tasks before returning them
        shuffle_seed: Seed for reproducible shuffling
    """

    # Tool configuration (required by Benchmark base class)
    tool_config: ComputerConfig = Field(default_factory=ComputerConfig)

    # Task selection
    test_set_path: str = str(OSWORLD_REPO_DIR / "evaluation_examples")
    test_set_name: str = "test_all.json"
    domain: str = "all"  # or specific domain like "chrome", "libreoffice", etc.
    shuffle: bool = True
    shuffle_seed: int = 42

    def setup(self) -> None:
        """Initialize benchmark.

        Automatically clones OSWorld repo if not present.
        Desktop_env VMs are managed per-task by the Computer tool.
        """
        logger.info(
            f"Setting up OSWorld benchmark with provider={self.tool_config.provider}, "
            f"domain={self.domain}"
        )

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

        Tasks are loaded from JSON files in the test_set_path directory.
        Each task specifies its configuration, evaluation criteria, and setup commands.

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
                        max_turns=50,
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
                parameters["settings_file"] = str(OSWORLD_REPO_DIR / settings_file)

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
                logger.info("✓ OSWorld repository cloned successfully")
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

    def uninstall(self) -> None:
        """Remove OSWorld resources.

        Removes downloaded VM images and cleans up installation.
        """
        logger.info("Uninstalling OSWorld...")
        logger.info("To remove OSWorld:")
        logger.info("1. Delete VM images")
        logger.info("2. Remove OSWorld repository")
        logger.info("3. Uninstall desktop_env: pip uninstall desktop-env")
