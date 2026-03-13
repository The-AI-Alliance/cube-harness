"""Tests for osworld_cube — verifies compliance with the CUBE protocol ABCs."""

from __future__ import annotations

import io
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

from PIL import Image

from cube.core import Action, Observation, TextContent

# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------

PATCH_QEMU_MGR = "osworld_cube.computer.QEMUManager"
PATCH_GUEST_AGENT = "osworld_cube.computer.GuestAgent"
PATCH_EVALUATOR = "osworld_cube.computer.Evaluator"
PATCH_SETUP_CTRL = "osworld_cube.computer.SetupController"
PATCH_ENSURE_IMAGE = "osworld_cube.computer.ensure_base_image"
PATCH_SLEEP = "osworld_cube.computer.time.sleep"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_screenshot_bytes(w: int = 100, h: int = 100) -> bytes:
    """Return a minimal PNG screenshot as bytes."""
    img = Image.new("RGB", (w, h), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_mock_qemu() -> MagicMock:
    """Return a Mock that looks like a started QEMUManager."""
    qemu = MagicMock()
    qemu.server_port = 15000
    qemu.chromium_port = 19222
    qemu.vnc_port = 18006
    qemu.vlc_port = 18080
    return qemu


def _make_mock_guest(screenshot: bytes | None = None, axtree: str = "<root/>") -> MagicMock:
    """Return a Mock that looks like GuestAgent."""
    guest = MagicMock()
    guest.get_screenshot.return_value = screenshot or _make_screenshot_bytes()
    guest.get_accessibility_tree.return_value = axtree
    guest.get_terminal_output.return_value = ""
    guest.execute_action.return_value = None
    guest.execute_python_command.return_value = {"returncode": 0, "output": ""}
    return guest


def _make_mock_evaluator(reward: float = 1.0) -> MagicMock:
    """Return a Mock that looks like Evaluator."""
    evaluator = MagicMock()
    evaluator.evaluate.return_value = reward
    return evaluator


@contextmanager
def _backend(
    screenshot: bytes | None = None,
    axtree: str = "<root/>",
    reward: float = 1.0,
) -> Generator[tuple[MagicMock, MagicMock, MagicMock], None, None]:
    """Context manager that patches all vm_backend components.

    Yields (mock_qemu, mock_guest, mock_evaluator).
    """
    mock_qemu = _make_mock_qemu()
    mock_guest = _make_mock_guest(screenshot, axtree)
    mock_evaluator = _make_mock_evaluator(reward)
    with (
        patch(PATCH_ENSURE_IMAGE, return_value=Path("/fake/Ubuntu.qcow2")),
        patch(PATCH_QEMU_MGR, return_value=mock_qemu),
        patch(PATCH_GUEST_AGENT, return_value=mock_guest),
        patch(PATCH_SETUP_CTRL),
        patch(PATCH_EVALUATOR, return_value=mock_evaluator),
    ):
        yield mock_qemu, mock_guest, mock_evaluator


# ---------------------------------------------------------------------------
# ComputerConfig
# ---------------------------------------------------------------------------


class TestComputerConfig:
    def test_defaults(self) -> None:
        from osworld_cube.computer import ComputerConfig
        from osworld_cube.vm_backend import VMConfig

        cfg = ComputerConfig()
        assert isinstance(cfg.vm_config, VMConfig)
        assert cfg.vm_config.screen_size == (1920, 1080)
        assert cfg.vm_config.headless is True
        assert cfg.require_a11y_tree is True
        assert cfg.observe_after_action is True
        assert cfg.vm_config.memory == "4G"
        assert cfg.vm_config.cpus == 4

    def test_custom_values(self) -> None:
        from osworld_cube.computer import ComputerConfig
        from osworld_cube.vm_backend import VMConfig

        cfg = ComputerConfig(vm_config=VMConfig(headless=False, screen_size=(1280, 720), memory="8G", cpus=8))
        assert cfg.vm_config.headless is False
        assert cfg.vm_config.screen_size == (1280, 720)
        assert cfg.vm_config.memory == "8G"
        assert cfg.vm_config.cpus == 8

    def test_action_space_default(self) -> None:
        from osworld_cube.computer import ActionSpace, ComputerConfig

        cfg = ComputerConfig()
        assert cfg.action_space == ActionSpace.COMPUTER_13


# ---------------------------------------------------------------------------
# Computer
# ---------------------------------------------------------------------------


class TestComputer:
    def test_init_starts_qemu(self) -> None:
        from osworld_cube.computer import ComputerConfig

        with _backend() as (mock_qemu, mock_guest, _):
            computer = ComputerConfig().make()
            mock_qemu.start.assert_called_once()
            assert computer.config is not None

    def test_action_set_computer13(self) -> None:
        from osworld_cube.computer import ComputerConfig

        with _backend():
            computer = ComputerConfig(action_space="computer_13").make()
            names = {a.name for a in computer.action_set}
            for expected in (
                "click",
                "double_click",
                "right_click",
                "drag_to",
                "scroll",
                "typing",
                "press",
                "hotkey",
                "wait",
                "done",
                "fail",
            ):
                assert expected in names, f"Missing action: {expected}"
            assert "run_pyautogui" not in names

    def test_action_set_pyautogui(self) -> None:
        from osworld_cube.computer import ComputerConfig

        with _backend():
            computer = ComputerConfig(action_space="pyautogui").make()
            names = {a.name for a in computer.action_set}
            assert "run_pyautogui" in names
            for terminal in ("wait", "done", "fail"):
                assert terminal in names, f"Missing action: {terminal}"
            assert "click" not in names

    def test_setup_task_returns_observation(self) -> None:
        from osworld_cube.computer import ComputerConfig

        with _backend() as (mock_qemu, _, _), patch(PATCH_SLEEP) as mock_sleep:
            computer = ComputerConfig().make()
            obs = computer.setup_task(
                {"id": "t1", "instruction": "test", "config": [], "evaluator": {}, "snapshot": "init_state"}
            )
            assert isinstance(obs, Observation)
            mock_qemu.reset.assert_called_once()
            mock_sleep.assert_called_once_with(60)

    def test_evaluate_task_returns_float(self) -> None:
        from osworld_cube.computer import ComputerConfig

        with _backend(reward=0.75) as (_, _, mock_evaluator):
            computer = ComputerConfig().make()
            computer._current_task_config = {"id": "t1", "evaluator": {}}
            reward = computer.evaluate_task()
            assert reward == 0.75

    def test_evaluate_task_returns_zero_on_exception(self) -> None:
        from osworld_cube.computer import ComputerConfig

        with _backend() as (_, _, mock_evaluator):
            mock_evaluator.evaluate.side_effect = RuntimeError("eval failed")
            computer = ComputerConfig().make()
            computer._current_task_config = {"id": "t1", "evaluator": {}}
            reward = computer.evaluate_task()
            assert reward == 0.0

    def test_click_dispatches_to_guest(self) -> None:
        from osworld_cube.computer import ComputerConfig

        with _backend() as (_, mock_guest, _):
            computer = ComputerConfig(observe_after_action=False).make()
            result = computer.click(x=100, y=200)
            assert result == "Success"
            mock_guest.execute_action.assert_called_once()
            call_args = mock_guest.execute_action.call_args[0][0]
            assert call_args["action_type"] == "CLICK"
            assert call_args["parameters"]["x"] == 100
            assert call_args["parameters"]["y"] == 200

    def test_done_sets_is_done(self) -> None:
        from osworld_cube.computer import ComputerConfig

        with _backend():
            computer = ComputerConfig(observe_after_action=False).make()
            assert computer._is_done is False
            computer.done()
            assert computer._is_done is True

    def test_fail_sets_is_done(self) -> None:
        from osworld_cube.computer import ComputerConfig

        with _backend():
            computer = ComputerConfig(observe_after_action=False).make()
            computer.fail()
            assert computer._is_done is True

    def test_update_marks_and_run_pyautogui(self) -> None:
        from osworld_cube.computer import ComputerConfig

        with _backend() as (_, mock_guest, _):
            computer = ComputerConfig(action_space="pyautogui", observe_after_action=False).make()
            computer.update_marks([[10, 20, 30, 40], [50, 60, 10, 10]])
            computer.run_pyautogui("pyautogui.click(*tag_1)")

            # tag_1 center: (10 + 30//2, 20 + 40//2) = (25, 40)
            # tag_2 center: (50 + 10//2, 60 + 10//2) = (55, 65)
            call_code = mock_guest.execute_python_command.call_args[0][0]
            assert "tag_1 = (25, 40)" in call_code
            assert "tag_2 = (55, 65)" in call_code
            assert "pyautogui.click(*tag_1)" in call_code

    def test_execute_action_dispatches_via_cube(self) -> None:
        """cube.tool.Tool.execute_action routes by action name to the correct @tool_action."""
        from osworld_cube.computer import ComputerConfig

        with _backend() as (_, mock_guest, _):
            computer = ComputerConfig(observe_after_action=False).make()
            result = computer.execute_action(Action(name="typing", arguments={"text": "hello"}))
            assert isinstance(result, Observation)
            mock_guest.execute_action.assert_called_once()

    def test_close_stops_qemu(self) -> None:
        from osworld_cube.computer import ComputerConfig

        with _backend() as (mock_qemu, _, _):
            computer = ComputerConfig().make()
            computer.close()
            mock_qemu.stop.assert_called_once()


# ---------------------------------------------------------------------------
# OSWorldTask
# ---------------------------------------------------------------------------


def _make_task_metadata(task_id: str = "t1", instruction: str = "Do something"):
    from cube.task import TaskMetadata

    return TaskMetadata(
        id=task_id,
        abstract_description=instruction,
        extra_info={
            "domain": "os",
            "snapshot": "init_state",
            "config": [],
            "evaluator": {"func": "check_file", "expected": {}},
            "related_apps": [],
        },
    )


class TestOSWorldTask:
    def test_reset_returns_obs_and_info(self) -> None:
        from osworld_cube.computer import ComputerConfig
        from osworld_cube.task import OSWorldTask

        with _backend(), patch(PATCH_SLEEP):
            task = OSWorldTask(
                metadata=_make_task_metadata(),
                tool_config=ComputerConfig(),
            )
            obs, info = task.reset()

            assert isinstance(obs, Observation)
            texts = [c.data for c in obs.contents if isinstance(c, TextContent)]
            assert any("Do something" in t for t in texts)
            assert info["task_id"] == "t1"
            assert info["task_domain"] == "os"

    def test_reset_resets_is_done(self) -> None:
        from osworld_cube.computer import ComputerConfig
        from osworld_cube.task import OSWorldTask

        with _backend(), patch(PATCH_SLEEP):
            task = OSWorldTask(
                metadata=_make_task_metadata(),
                tool_config=ComputerConfig(),
            )
            task._computer._is_done = True
            task.reset()
            assert task._computer._is_done is False

    def test_evaluate_returns_reward_and_info(self) -> None:
        from osworld_cube.computer import ComputerConfig
        from osworld_cube.task import OSWorldTask

        with _backend(reward=0.5):
            task = OSWorldTask(
                metadata=_make_task_metadata(),
                tool_config=ComputerConfig(),
            )
            task._computer._current_task_config = {"id": "t1", "evaluator": {"func": "check_file"}}
            obs = Observation.from_text("state")
            reward, info = task.evaluate(obs)

            assert reward == 0.5
            assert "evaluator" in info

    def test_evaluate_no_evaluator_returns_zero(self) -> None:
        from cube.task import TaskMetadata

        from osworld_cube.computer import ComputerConfig
        from osworld_cube.task import OSWorldTask

        with _backend():
            task = OSWorldTask(
                metadata=TaskMetadata(id="no-eval", extra_info={}),
                tool_config=ComputerConfig(),
            )
            reward, info = task.evaluate(Observation())
            assert reward == 0.0
            assert info.get("error") == "no_evaluator"

    def test_finished_reflects_computer_is_done(self) -> None:
        from osworld_cube.computer import ComputerConfig
        from osworld_cube.task import OSWorldTask

        with _backend():
            task = OSWorldTask(
                metadata=_make_task_metadata(),
                tool_config=ComputerConfig(),
            )
            assert task.finished(Observation()) is False
            task._computer._is_done = True
            assert task.finished(Observation()) is True

    def test_step_done_action_triggers_evaluate(self) -> None:
        """
        Full step loop: agent calls done() → task.step() sets done=True →
        evaluate() is called → EnvironmentOutput.done is True.
        """
        from osworld_cube.computer import ComputerConfig
        from osworld_cube.task import OSWorldTask

        with _backend(reward=1.0), patch(PATCH_SLEEP):
            task = OSWorldTask(
                metadata=_make_task_metadata(),
                tool_config=ComputerConfig(observe_after_action=False),
            )
            task.reset()

            env_out = task.step(Action(name="done", arguments={}))

            assert env_out.done is True
            assert env_out.reward == 1.0

    def test_step_click_not_done(self) -> None:
        """A regular action does not set done."""
        from osworld_cube.computer import ComputerConfig
        from osworld_cube.task import OSWorldTask

        with _backend(), patch(PATCH_SLEEP):
            task = OSWorldTask(
                metadata=_make_task_metadata(),
                tool_config=ComputerConfig(observe_after_action=False),
            )
            task.reset()

            env_out = task.step(Action(name="click", arguments={"x": 10, "y": 20}))
            assert env_out.done is False

    def test_close_stops_qemu(self) -> None:
        from osworld_cube.computer import ComputerConfig
        from osworld_cube.task import OSWorldTask

        with _backend() as (mock_qemu, _, _), patch(PATCH_SLEEP):
            task = OSWorldTask(
                metadata=_make_task_metadata(),
                tool_config=ComputerConfig(),
            )
            task.reset()
            task.close()
            mock_qemu.stop.assert_called_once()


# ---------------------------------------------------------------------------
# OSWorldTestSet
# ---------------------------------------------------------------------------


class TestOSWorldTestSet:
    def test_enum_values_are_filenames(self) -> None:
        from osworld_cube.benchmark import OSWorldTestSet

        assert OSWorldTestSet.TEST_ALL.value == "test_all.json"
        assert OSWorldTestSet.TEST_INFEASIBLE.value == "test_infeasible.json"
        assert OSWorldTestSet.TEST_NOGDRIVE.value == "test_nogdrive.json"
        assert OSWorldTestSet.TEST_SMALL.value == "test_small.json"

    def test_enum_is_str_subclass(self) -> None:
        from osworld_cube.benchmark import OSWorldTestSet

        assert isinstance(OSWorldTestSet.TEST_ALL, str)
        assert OSWorldTestSet.TEST_ALL == "test_all.json"

    def test_enum_from_string(self) -> None:
        from osworld_cube.benchmark import OSWorldTestSet

        assert OSWorldTestSet("test_small.json") is OSWorldTestSet.TEST_SMALL


# ---------------------------------------------------------------------------
# OSWorldBenchmark
# ---------------------------------------------------------------------------


def _make_osworld_repo(tmpdir: Path) -> Path:
    """Create a minimal fake OSWorld repo with 2 tasks in 2 domains."""
    eval_dir = tmpdir / "evaluation_examples"
    (eval_dir / "examples" / "chrome").mkdir(parents=True)
    (eval_dir / "examples" / "os").mkdir(parents=True)

    test_set = {"chrome": ["chrome-1"], "os": ["os-1"]}
    (eval_dir / "test_all.json").write_text(json.dumps(test_set))

    (eval_dir / "examples" / "chrome" / "chrome-1.json").write_text(
        json.dumps(
            {
                "id": "chrome-1",
                "instruction": "Open Chrome",
                "snapshot": "init_state",
                "config": [],
                "evaluator": {"func": "check_url"},
                "related_apps": ["chrome"],
            }
        )
    )
    (eval_dir / "examples" / "os" / "os-1.json").write_text(
        json.dumps(
            {
                "id": "os-1",
                "instruction": "Open terminal",
                "snapshot": "init_state",
                "config": [],
                "evaluator": {"func": "check_process"},
                "related_apps": [],
            }
        )
    )
    return eval_dir


class TestOSWorldBenchmark:
    def test_benchmark_metadata(self) -> None:
        from osworld_cube.benchmark import OSWorldBenchmark

        assert OSWorldBenchmark.benchmark_metadata.name == "osworld"
        assert OSWorldBenchmark.task_config_class.__name__ == "OSWorldTaskConfig"

    def test_load_all_tasks_from_repo(self) -> None:
        from osworld_cube.benchmark import OSWorldBenchmark
        from osworld_cube.computer import ComputerConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            eval_dir = _make_osworld_repo(Path(tmpdir))
            bench = OSWorldBenchmark(
                default_tool_config=ComputerConfig(),
                test_set_path=str(eval_dir),
                test_set_name="test_all.json",
            )
            bench.setup()

            assert len(bench.task_metadata) == 2
            assert "chrome-1" in bench.task_metadata
            assert "os-1" in bench.task_metadata

    def test_domain_filter_via_subset_from_glob(self) -> None:
        from osworld_cube.benchmark import OSWorldBenchmark
        from osworld_cube.computer import ComputerConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            eval_dir = _make_osworld_repo(Path(tmpdir))
            bench = OSWorldBenchmark(
                default_tool_config=ComputerConfig(),
                test_set_path=str(eval_dir),
                test_set_name="test_all.json",
            )
            bench.setup()
            chrome_bench = bench.subset_from_glob("extra_info.domain", "chrome")

            assert len(chrome_bench.task_metadata) == 1
            assert "chrome-1" in chrome_bench.task_metadata

    def test_get_task_configs_carries_metadata(self) -> None:
        from osworld_cube.benchmark import OSWorldBenchmark, OSWorldTaskConfig
        from osworld_cube.computer import ComputerConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            eval_dir = _make_osworld_repo(Path(tmpdir))
            bench = OSWorldBenchmark(
                default_tool_config=ComputerConfig(),
                test_set_path=str(eval_dir),
            )
            bench.setup()

            configs = list(bench.get_task_configs())
            assert len(configs) == 2
            for cfg in configs:
                assert isinstance(cfg, OSWorldTaskConfig)
                assert cfg.metadata is not None
                assert cfg.task_id == cfg.metadata.id

    def test_task_config_make_produces_osworld_task(self) -> None:
        from osworld_cube.benchmark import OSWorldBenchmark
        from osworld_cube.computer import ComputerConfig
        from osworld_cube.task import OSWorldTask

        with tempfile.TemporaryDirectory() as tmpdir:
            eval_dir = _make_osworld_repo(Path(tmpdir))
            bench = OSWorldBenchmark(
                default_tool_config=ComputerConfig(),
                test_set_path=str(eval_dir),
            )
            bench.setup()

            cfg = next(bench.get_task_configs())

            with _backend():
                task = cfg.make()

            assert isinstance(task, OSWorldTask)
            assert task.metadata.id == cfg.task_id

    def test_load_from_flat_json_file(self) -> None:
        from osworld_cube.benchmark import OSWorldBenchmark
        from osworld_cube.computer import ComputerConfig

        tasks_data = [
            {
                "id": "flat-1",
                "instruction": "Flat task 1",
                "domain": "os",
                "snapshot": "init_state",
                "config": [],
                "evaluator": {},
                "related_apps": [],
            },
            {
                "id": "flat-2",
                "instruction": "Flat task 2",
                "domain": "chrome",
                "snapshot": "init_state",
                "config": [],
                "evaluator": {},
                "related_apps": [],
            },
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(tasks_data, f)
            tasks_file = f.name

        bench = OSWorldBenchmark(
            default_tool_config=ComputerConfig(),
            tasks_file=tasks_file,
        )
        bench.setup()

        assert len(bench.task_metadata) == 2
        assert bench.task_metadata["flat-1"].abstract_description == "Flat task 1"
        assert bench.task_metadata["flat-2"].extra_info["domain"] == "chrome"

    def test_fix_settings_paths(self) -> None:
        import os

        from osworld_cube.benchmark import OSWorldBenchmark
        from osworld_cube.computer import ComputerConfig

        bench = OSWorldBenchmark(default_tool_config=ComputerConfig())
        task_data = {
            "id": "t",
            "config": [{"type": "setup", "parameters": {"settings_file": "configs/x.json"}}],
        }
        with patch.dict(os.environ, {"OSWORLD_REPO": "/fake/osworld"}):
            fixed = bench._fix_settings_paths(task_data)

        assert fixed["config"][0]["parameters"]["settings_file"] == "/fake/osworld/configs/x.json"

    def test_test_set_name_accepts_enum(self) -> None:
        from osworld_cube.benchmark import OSWorldBenchmark, OSWorldTestSet
        from osworld_cube.computer import ComputerConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            eval_dir = _make_osworld_repo(Path(tmpdir))
            bench = OSWorldBenchmark(
                default_tool_config=ComputerConfig(),
                test_set_path=str(eval_dir),
                test_set_name=OSWorldTestSet.TEST_ALL,
            )
            bench.setup()
            assert len(bench.task_metadata) == 2

    def test_test_set_name_selects_subset(self) -> None:
        """Different enum values load different task subsets from the repo."""
        from osworld_cube.benchmark import OSWorldBenchmark, OSWorldTestSet
        from osworld_cube.computer import ComputerConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            eval_dir = _make_osworld_repo(Path(tmpdir))
            (eval_dir / "test_small.json").write_text(json.dumps({"chrome": ["chrome-1"]}))

            bench_all = OSWorldBenchmark(
                default_tool_config=ComputerConfig(),
                test_set_path=str(eval_dir),
                test_set_name=OSWorldTestSet.TEST_ALL,
            )
            bench_small = OSWorldBenchmark(
                default_tool_config=ComputerConfig(),
                test_set_path=str(eval_dir),
                test_set_name=OSWorldTestSet.TEST_SMALL,
            )
            bench_all.setup()
            bench_small.setup()

            assert len(bench_all.task_metadata) == 2
            assert len(bench_small.task_metadata) == 1
            assert "chrome-1" in bench_small.task_metadata
            assert "os-1" not in bench_small.task_metadata

    def test_close_does_not_raise(self) -> None:
        from osworld_cube.benchmark import OSWorldBenchmark
        from osworld_cube.computer import ComputerConfig

        OSWorldBenchmark(default_tool_config=ComputerConfig()).close()
