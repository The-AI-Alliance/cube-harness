"""Integration tests for OSWorld implementation.

These tests verify the OSWorld benchmark, tasks, and Computer tool work correctly.
Tests use mocked desktop_env to avoid requiring actual VM setup.
"""

import io
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from agentlab2.action_spaces.computer_action_space import ComputerActionSpace
from agentlab2.benchmarks.osworld.benchmark import OSWorldBenchmark
from agentlab2.benchmarks.osworld.task import OSWorldTask
from agentlab2.core import Observation
from agentlab2.tools.computer import Computer, ComputerConfig


class TestComputerActionSpace:
    """Test ComputerActionSpace protocol definition."""

    def test_action_space_protocol_exists(self):
        """Test that ComputerActionSpace protocol is defined."""
        assert hasattr(ComputerActionSpace, "__annotations__")

    def test_action_space_has_all_methods(self):
        """Test that all 15 Computer_13 actions are defined."""
        expected_methods = [
            "move_to",
            "click",
            "mouse_down",
            "mouse_up",
            "right_click",
            "double_click",
            "drag_to",
            "scroll",
            "typing",
            "press",
            "key_down",
            "key_up",
            "hotkey",
            "wait",
            "fail",
            "done",
        ]

        # Check protocol has all methods
        for method in expected_methods:
            assert hasattr(ComputerActionSpace, method)


class TestComputerConfig:
    """Test ComputerConfig class."""

    def test_config_creation(self):
        """Test creating ComputerConfig with default values."""
        config = ComputerConfig()
        assert config.provider == "docker"
        assert config.screen_size == (1920, 1080)
        assert config.headless == True
        assert config.require_a11y_tree == True

    def test_config_custom_values(self):
        """Test creating ComputerConfig with custom values."""
        config = ComputerConfig(
            provider="vmware",
            screen_size=(1280, 720),
            headless=False,
            require_a11y_tree=False,
        )
        assert config.provider == "vmware"
        assert config.screen_size == (1280, 720)
        assert config.headless == False
        assert config.require_a11y_tree == False


class TestComputer:
    """Test Computer tool with mocked desktop_env."""

    @patch("agentlab2.tools.computer.DesktopEnv")
    def test_computer_initialization(self, mock_desktop_env):
        """Test Computer tool initializes correctly."""
        config = ComputerConfig()
        computer = config.make()

        assert isinstance(computer, Computer)
        assert computer.config == config
        assert mock_desktop_env.called

    @patch("agentlab2.tools.computer.DesktopEnv", None)
    def test_computer_requires_desktop_env(self):
        """Test that Computer raises ImportError without desktop_env."""
        config = ComputerConfig()
        with pytest.raises(ImportError, match="desktop_env is not installed"):
            config.make()

    @patch("agentlab2.tools.computer.DesktopEnv")
    def test_computer_action_execution(self, mock_desktop_env):
        """Test executing actions through Computer tool."""
        # Setup mock
        mock_env = Mock()
        mock_env.step.return_value = ({}, 0.0, False, {})
        mock_desktop_env.return_value = mock_env

        config = ComputerConfig()
        computer = config.make()

        # Test move_to action
        result = computer.move_to(100, 200)
        assert result == "Success"
        mock_env.step.assert_called_with(
            {"action_type": "MOVE_TO", "parameters": {"x": 100, "y": 200}}
        )

        # Test click action
        result = computer.click()
        assert result == "Success"
        assert mock_env.step.called

        # Test typing action
        result = computer.typing("hello world")
        assert result == "Success"
        assert mock_env.step.called

    @patch("agentlab2.tools.computer.DesktopEnv")
    def test_computer_setup_task(self, mock_desktop_env):
        """Test Computer tool setup_task method."""
        # Create a real minimal PIL image
        test_img = Image.new("RGB", (100, 100), color="white")
        img_bytes = io.BytesIO()
        test_img.save(img_bytes, format="PNG")
        img_bytes = img_bytes.getvalue()

        # Setup mock
        mock_env = Mock()
        mock_env.reset.return_value = None
        mock_env._get_obs.return_value = {
            "screenshot": img_bytes,
            "accessibility_tree": "<root></root>",
        }
        mock_desktop_env.return_value = mock_env

        config = ComputerConfig()
        computer = config.make()

        task_config = {
            "id": "test-task",
            "instruction": "Test instruction",
            "config": [],
            "evaluator": {},
            "snapshot": "init_state",
        }

        # Mock the sleep to speed up test
        with patch("time.sleep"):
            obs = computer.setup_task(task_config)

        assert isinstance(obs, Observation)
        mock_env.reset.assert_called_once()
        mock_env._get_obs.assert_called_once()

    @patch("agentlab2.tools.computer.DesktopEnv")
    def test_computer_evaluate_task(self, mock_desktop_env):
        """Test Computer tool evaluate_task method."""
        # Setup mock
        mock_env = Mock()
        mock_env.evaluate.return_value = 1.0
        mock_desktop_env.return_value = mock_env

        config = ComputerConfig()
        computer = config.make()

        reward = computer.evaluate_task()
        assert reward == 1.0
        mock_env.evaluate.assert_called_once()


class TestOSWorldTask:
    """Test OSWorldTask class."""

    def test_task_creation(self):
        """Test creating an OSWorldTask."""
        task = OSWorldTask(
            id="test-task-123",
            desc="Test task description",
            domain="chrome",
            instruction="Open Chrome and navigate to google.com",
            snapshot="init_state",
            related_apps=["chrome"],
            config=[],
            evaluator={"func": "check_url", "expected": {"url": "google.com"}},
        )

        assert task.id == "test-task-123"
        assert task.domain == "chrome"
        assert task.instruction == "Open Chrome and navigate to google.com"
        assert task.snapshot == "init_state"
        assert len(task.related_apps) == 1
        assert task.evaluator["func"] == "check_url"

    @patch("agentlab2.tools.computer.DesktopEnv")
    def test_task_setup(self, mock_desktop_env):
        """Test task setup method."""
        # Create a real minimal PIL image
        test_img = Image.new("RGB", (100, 100), color="white")
        img_bytes = io.BytesIO()
        test_img.save(img_bytes, format="PNG")
        img_bytes = img_bytes.getvalue()

        # Setup mock tool
        mock_env = Mock()
        mock_env.reset.return_value = None
        mock_env._get_obs.return_value = {
            "screenshot": img_bytes,
            "accessibility_tree": "<root></root>",
        }
        mock_desktop_env.return_value = mock_env

        config = ComputerConfig()
        tool = config.make()

        # Create task
        task = OSWorldTask(
            id="test-task",
            desc="Test",
            instruction="Do something",
            evaluator={"func": "test", "expected": {}},
        )

        # Setup task
        with patch("time.sleep"):
            obs, info = task.setup(tool)

        assert isinstance(obs, Observation)
        assert info["task_id"] == "test-task"
        assert info["task_domain"] == "general"
        assert task._tool == tool
        assert task._is_done == False

    @patch("agentlab2.tools.computer.DesktopEnv")
    def test_task_validate(self, mock_desktop_env):
        """Test task validation method."""
        # Setup mock tool
        mock_env = Mock()
        mock_env.evaluate.return_value = 1.0
        mock_desktop_env.return_value = mock_env

        config = ComputerConfig()
        tool = config.make()

        # Create and setup task
        task = OSWorldTask(
            id="test-task",
            desc="Test",
            evaluator={"func": "test", "expected": {}},
        )
        task._tool = tool

        # Validate task
        obs = Observation.from_text("test")
        reward, info = task.validate_task(obs)

        assert reward == 1.0
        assert info["done"] == True
        assert info["evaluator"] == "test"

    def test_task_finished(self):
        """Test task finished state."""
        task = OSWorldTask(id="test", desc="Test", evaluator={})

        assert task.finished() == False

        task.mark_done(success=True)
        assert task.finished() == True


class TestOSWorldBenchmark:
    """Test OSWorldBenchmark class."""

    def test_benchmark_creation(self):
        """Test creating an OSWorldBenchmark."""
        benchmark = OSWorldBenchmark(
            tool_config=ComputerConfig(),
            domain="chrome",
            shuffle=False,
        )

        assert isinstance(benchmark.tool_config, ComputerConfig)
        assert benchmark.domain == "chrome"
        assert benchmark.shuffle == False

    def test_benchmark_setup(self):
        """Test benchmark setup method."""
        benchmark = OSWorldBenchmark()
        # Should not raise any errors even if files don't exist
        benchmark.setup()

    def test_benchmark_load_tasks_from_json(self):
        """Test loading tasks from JSON files."""
        # Create temporary test set files
        with tempfile.TemporaryDirectory() as tmpdir:
            test_set_path = Path(tmpdir)

            # Create test set index
            test_set_file = test_set_path / "test_small.json"
            test_set_data = {"test_domain": ["task-1", "task-2"]}
            with open(test_set_file, "w") as f:
                json.dump(test_set_data, f)

            # Create examples directory
            examples_dir = test_set_path / "examples" / "test_domain"
            examples_dir.mkdir(parents=True)

            # Create task files
            task1_file = examples_dir / "task-1.json"
            task1_data = {
                "id": "task-1",
                "instruction": "Test task 1",
                "snapshot": "init_state",
                "related_apps": [],
                "config": [],
                "evaluator": {"func": "test"},
            }
            with open(task1_file, "w") as f:
                json.dump(task1_data, f)

            task2_file = examples_dir / "task-2.json"
            task2_data = {
                "id": "task-2",
                "instruction": "Test task 2",
                "snapshot": "init_state",
                "related_apps": [],
                "config": [],
                "evaluator": {"func": "test"},
            }
            with open(task2_file, "w") as f:
                json.dump(task2_data, f)

            # Create benchmark and load tasks
            benchmark = OSWorldBenchmark(
                test_set_path=str(test_set_path),
                test_set_name="test_small.json",
                domain="all",
                shuffle=False,
            )
            tasks = benchmark.load_tasks()

            assert len(tasks) == 2
            assert tasks[0].id == "task-1"
            assert tasks[1].id == "task-2"
            assert tasks[0].instruction == "Test task 1"
            assert tasks[1].instruction == "Test task 2"

    def test_benchmark_domain_filtering(self):
        """Test domain filtering in task loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_set_path = Path(tmpdir)

            # Create test set with multiple domains
            test_set_file = test_set_path / "test.json"
            test_set_data = {
                "chrome": ["chrome-1"],
                "os": ["os-1"],
            }
            with open(test_set_file, "w") as f:
                json.dump(test_set_data, f)

            # Create task files
            for domain, task_ids in test_set_data.items():
                domain_dir = test_set_path / "examples" / domain
                domain_dir.mkdir(parents=True)
                for task_id in task_ids:
                    task_file = domain_dir / f"{task_id}.json"
                    task_data = {
                        "id": task_id,
                        "instruction": f"Task {task_id}",
                        "snapshot": "init_state",
                        "related_apps": [],
                        "config": [],
                        "evaluator": {},
                    }
                    with open(task_file, "w") as f:
                        json.dump(task_data, f)

            # Load only chrome domain
            benchmark = OSWorldBenchmark(
                test_set_path=str(test_set_path),
                test_set_name="test.json",
                domain="chrome",
                shuffle=False,
            )
            tasks = benchmark.load_tasks()

            assert len(tasks) == 1
            assert tasks[0].id == "chrome-1"

    def test_benchmark_path_fixing(self):
        """Test _fix_settings_paths method."""
        benchmark = OSWorldBenchmark()

        task_data = {
            "id": "test",
            "config": [
                {
                    "type": "setup",
                    "parameters": {"settings_file": "configs/chrome_settings.json"},
                }
            ],
        }

        with patch.dict("os.environ", {"OSWORLD_REPO": "/path/to/osworld"}):
            fixed_task = benchmark._fix_settings_paths(task_data)

        assert fixed_task["config"][0]["parameters"]["settings_file"] == (
            "/path/to/osworld/configs/chrome_settings.json"
        )

    def test_benchmark_close(self):
        """Test benchmark close method."""
        benchmark = OSWorldBenchmark()
        # Should not raise any errors
        benchmark.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
