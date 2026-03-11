"""Tests for OSWorld benchmark, tasks, and Computer tool.

Uses mocked LocalQEMUVMBackend and HTTP requests to avoid requiring actual VM setup.
"""

import io
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from agentlab2.core import Observation as AL2Observation
from cube.core import Observation as CubeObservation
from cube.vm import VMConfig
from cube_computer_tool import Computer, ComputerConfig
from cube_computer_tool.backends.local_qemu import LocalQEMUVMBackend
from osworld_cube import OSWorldBenchmark, OSWorldComputerConfig, OSWorldTask


def _make_png_bytes(width: int = 100, height: int = 100, color: str = "white") -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_mock_vm() -> Mock:
    vm = Mock()
    vm.endpoint = "http://localhost:5000"
    vm.restore_snapshot = Mock()
    vm.stop = Mock()
    return vm


def _make_computer(require_a11y_tree: bool = True, observe_after_action: bool = True):
    """Create a Computer with a mocked VM, bypassing pydantic validation."""
    mock_vm = _make_mock_vm()
    config = ComputerConfig.model_construct(
        vm_backend=None,
        vm_config=VMConfig(),
        require_a11y_tree=require_a11y_tree,
        require_terminal=False,
        observe_after_action=observe_after_action,
    )
    return Computer(config=config, vm=mock_vm), mock_vm


def _action_names_from_class(cls) -> list[str]:
    """Inspect a class MRO and return names of all @tool_action decorated methods."""
    names = []
    for attr_name in dir(cls):
        if attr_name.startswith("_"):
            continue
        for klass in cls.__mro__:
            if attr_name in klass.__dict__:
                if getattr(klass.__dict__[attr_name], "_is_action", False):
                    names.append(attr_name)
                break
    return names


class TestComputerActions:
    """Test that Computer exposes all expected action methods."""

    def test_computer_has_all_actions(self):
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
            "done",
            "fail",
        ]
        for method in expected_methods:
            assert hasattr(Computer, method), f"Computer is missing action: {method}"

    def test_computer_tool_action_decorations(self):
        """All expected methods are decorated with @tool_action."""
        action_names = _action_names_from_class(Computer)
        assert "click" in action_names
        assert "typing" in action_names
        assert "done" in action_names
        assert "fail" in action_names
        assert len(action_names) >= 16


class TestOSWorldComputerConfig:
    """Test OSWorldComputerConfig class."""

    def test_config_default_values(self):
        config = OSWorldComputerConfig()
        assert config.headless is True
        assert config.require_a11y_tree is True
        assert config.observe_after_action is True
        assert config.vm_image_path is None

    def test_config_custom_values(self):
        config = OSWorldComputerConfig(
            headless=False,
            require_a11y_tree=False,
            vm_image_path="/tmp/test.qcow2",
        )
        assert config.headless is False
        assert config.require_a11y_tree is False
        assert config.vm_image_path == "/tmp/test.qcow2"

    @patch("osworld_cube.config.get_osworld_vm_image")
    @patch("osworld_cube.config.ComputerConfig")
    def test_make_downloads_image_when_not_set(self, mock_computer_config_cls, mock_get_image):
        """make() auto-downloads qcow2 when vm_image_path is None."""
        mock_get_image.return_value = "/tmp/auto.qcow2"
        mock_computer_config = Mock()
        mock_computer_config.make.return_value = Mock(spec=Computer)
        mock_computer_config_cls.return_value = mock_computer_config

        config = OSWorldComputerConfig()
        config.make()

        mock_get_image.assert_called_once()

    @patch("osworld_cube.config.get_osworld_vm_image")
    @patch("osworld_cube.config.ComputerConfig")
    def test_make_uses_provided_image_path(self, mock_computer_config_cls, mock_get_image):
        """make() uses vm_image_path when explicitly provided (no download)."""
        mock_computer_config = Mock()
        mock_computer_config.make.return_value = Mock(spec=Computer)
        mock_computer_config_cls.return_value = mock_computer_config

        config = OSWorldComputerConfig(vm_image_path="/my/custom.qcow2")
        config.make()

        mock_get_image.assert_not_called()
        call_kwargs = mock_computer_config_cls.call_args.kwargs
        assert call_kwargs["vm_backend"].vm_image_path == "/my/custom.qcow2"


class TestComputer:
    """Test Computer tool with mocked VM and HTTP requests."""

    @patch("cube_computer_tool.computer.requests.get")
    def test_setup_task(self, mock_get):
        """setup_task() restores snapshot, runs setup, returns Observation."""
        png_bytes = _make_png_bytes()
        mock_screenshot = Mock()
        mock_screenshot.content = png_bytes
        mock_screenshot.raise_for_status = Mock()
        mock_axtree = Mock()
        mock_axtree.json.return_value = {"AT": "<root/>"}
        mock_axtree.raise_for_status = Mock()
        mock_get.side_effect = [mock_screenshot, mock_axtree]

        computer, mock_vm = _make_computer()
        task_config = {
            "id": "test-task",
            "instruction": "Do something",
            "config": [],
            "snapshot": "init_state",
        }

        with patch("time.sleep"):
            obs = computer.setup_task(task_config)

        assert isinstance(obs, CubeObservation)
        mock_vm.restore_snapshot.assert_called_once_with("init_state")

    @patch("cube_computer_tool.computer.requests.post")
    def test_click_action(self, mock_post):
        """click() sends PyAutoGUI code to POST /execute."""
        mock_post.return_value = Mock(
            status_code=200,
            json=Mock(return_value={"returncode": 0}),
        )
        mock_post.return_value.raise_for_status = Mock()

        computer, _ = _make_computer(observe_after_action=False)
        result = computer.click(x=100, y=200)

        assert result == "Success"
        call_args = mock_post.call_args
        payload = json.loads(call_args.kwargs.get("data") or call_args.args[1])
        assert "pyautogui.click" in payload["command"][2]
        assert "100" in payload["command"][2]
        assert "200" in payload["command"][2]

    @patch("cube_computer_tool.computer.requests.post")
    def test_typing_action(self, mock_post):
        """typing() sends PyAutoGUI typewrite code to POST /execute."""
        mock_post.return_value = Mock(
            status_code=200,
            json=Mock(return_value={"returncode": 0}),
        )
        mock_post.return_value.raise_for_status = Mock()

        computer, _ = _make_computer(observe_after_action=False)
        result = computer.typing("hello world")

        assert result == "Success"
        call_args = mock_post.call_args
        payload = json.loads(call_args.kwargs.get("data") or call_args.args[1])
        assert "typewrite" in payload["command"][2]
        assert "hello world" in payload["command"][2]

    def test_done_action_sets_is_done(self):
        """done() sets _is_done flag."""
        computer, _ = _make_computer()
        result = computer.done()
        assert computer._is_done is True
        assert "done" in result.lower()

    def test_fail_action_sets_is_done(self):
        """fail() sets _is_done flag."""
        computer, _ = _make_computer()
        computer.fail()
        assert computer._is_done is True

    def test_evaluate_task_returns_zero(self):
        """evaluate_task() returns 0.0 placeholder."""
        computer, _ = _make_computer()
        reward = computer.evaluate_task()
        assert reward == 0.0

    def test_close_stops_vm(self):
        """close() calls vm.stop()."""
        computer, mock_vm = _make_computer()
        computer.close()
        mock_vm.stop.assert_called_once()


class TestOSWorldTask:
    """Test OSWorldTask class."""

    def test_task_creation(self):
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

    def test_task_setup(self):
        """task.setup() calls tool.setup_task() and returns (Observation, info)."""
        mock_tool = Mock()
        mock_tool.setup_task.return_value = AL2Observation.from_text("screen")

        task = OSWorldTask(
            id="test-task",
            desc="Test",
            instruction="Do something",
            evaluator={"func": "test", "expected": {}},
        )
        result_obs, info = task.setup(mock_tool)

        mock_tool.setup_task.assert_called_once()
        assert isinstance(result_obs, AL2Observation)
        assert info["task_id"] == "test-task"
        assert info["task_domain"] == "general"
        assert task._tool is mock_tool
        assert task._is_done is False

    def test_task_validate_calls_evaluate(self):
        """validate_task() delegates to tool.evaluate_task()."""
        mock_tool = Mock()
        mock_tool.evaluate_task.return_value = 1.0

        task = OSWorldTask(
            id="test-task",
            desc="Test",
            evaluator={"func": "check_something", "expected": {}},
        )
        task._tool = mock_tool

        obs = AL2Observation.from_text("test")
        reward, info = task.validate_task(obs)

        assert reward == 1.0
        assert info["done"] is True
        assert info["evaluator"] == "check_something"

    def test_task_finished_and_mark_done(self):
        task = OSWorldTask(id="test", desc="Test", evaluator={})
        assert task.finished() is False

        task.mark_done(success=True)
        assert task._is_done is True
        assert task.finished() is True


class TestOSWorldBenchmark:
    """Test OSWorldBenchmark class."""

    def test_benchmark_creation(self):
        benchmark = OSWorldBenchmark(
            tool_config=OSWorldComputerConfig(),
            domain="chrome",
            shuffle=False,
        )
        assert isinstance(benchmark.tool_config, OSWorldComputerConfig)
        assert benchmark.domain == "chrome"
        assert benchmark.shuffle is False

    def test_benchmark_close_no_error(self):
        benchmark = OSWorldBenchmark()
        benchmark.close()

    def test_benchmark_load_tasks_from_repo_structure(self):
        """load_tasks() reads from OSWorld test set directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_set_path = Path(tmpdir)

            test_set_file = test_set_path / "test_small.json"
            test_set_data = {"test_domain": ["task-1", "task-2"]}
            with open(test_set_file, "w") as f:
                json.dump(test_set_data, f)

            examples_dir = test_set_path / "examples" / "test_domain"
            examples_dir.mkdir(parents=True)
            for task_id in ["task-1", "task-2"]:
                task_data = {
                    "id": task_id,
                    "instruction": f"Task {task_id}",
                    "snapshot": "init_state",
                    "related_apps": [],
                    "config": [],
                    "evaluator": {"func": "test"},
                }
                with open(examples_dir / f"{task_id}.json", "w") as f:
                    json.dump(task_data, f)

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
        assert tasks[0].instruction == "Task task-1"

    def test_benchmark_domain_filtering(self):
        """load_tasks() filters to the requested domain."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_set_path = Path(tmpdir)

            test_set_file = test_set_path / "test.json"
            test_set_data = {"chrome": ["chrome-1"], "os": ["os-1"]}
            with open(test_set_file, "w") as f:
                json.dump(test_set_data, f)

            for domain, task_ids in test_set_data.items():
                domain_dir = test_set_path / "examples" / domain
                domain_dir.mkdir(parents=True)
                for task_id in task_ids:
                    task_data = {
                        "id": task_id,
                        "instruction": f"Task {task_id}",
                        "snapshot": "init_state",
                        "related_apps": [],
                        "config": [],
                        "evaluator": {},
                    }
                    with open(domain_dir / f"{task_id}.json", "w") as f:
                        json.dump(task_data, f)

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
        """_fix_settings_paths() prepends OSWORLD_REPO to relative settings_file paths."""
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
