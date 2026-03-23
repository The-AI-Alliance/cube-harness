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
# Patch targets — pointing at cube_computer_tool and osworld_cube.task
# ---------------------------------------------------------------------------

PATCH_QEMU_MGR = "cube_vm_backend.local.QEMUManager"
PATCH_GUEST_AGENT = "cube_computer_tool.computer.GuestAgent"
PATCH_EVALUATOR = "osworld_cube.task.Evaluator"
PATCH_SETUP_CTRL = "osworld_cube.task.SetupController"
PATCH_ENSURE_IMAGE = "cube_vm_backend.local.ensure_base_image"
PATCH_SLEEP = "osworld_cube.task.time.sleep"

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

        cfg = ComputerConfig()
        assert cfg.require_a11y_tree is True
        assert cfg.observe_after_action is True

    def test_action_space_default(self) -> None:
        from osworld_cube.computer import ActionSpace, ComputerConfig

        cfg = ComputerConfig()
        assert cfg.action_space == ActionSpace.COMPUTER_13


# ---------------------------------------------------------------------------
# Computer
# ---------------------------------------------------------------------------


class TestComputer:
    def test_make_without_vm_succeeds(self) -> None:
        """ComputerConfig.make() with no vm creates a tool in unattached state."""
        from osworld_cube.computer import ComputerConfig

        cfg = ComputerConfig()
        computer = cfg.make()
        assert computer.config is not None
        assert computer._vm is None
        assert computer._guest is None

    def test_action_set_computer13(self) -> None:
        from osworld_cube.computer import ComputerConfig

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

        computer = ComputerConfig(action_space="pyautogui").make()
        names = {a.name for a in computer.action_set}
        assert "run_pyautogui" in names
        for terminal in ("wait", "done", "fail"):
            assert terminal in names, f"Missing action: {terminal}"
        assert "click" not in names

    def test_attach_vm_connects_guest(self) -> None:
        from osworld_cube.computer import ComputerConfig

        computer = ComputerConfig().make()
        mock_vm = MagicMock()
        mock_vm.endpoint = "http://localhost:15000"

        with patch(PATCH_GUEST_AGENT) as mock_ga_cls:
            mock_ga_cls.return_value = MagicMock()
            computer.attach_vm(mock_vm)

        assert computer._vm is mock_vm
        assert computer._guest is not None

    def test_click_dispatches_to_guest(self) -> None:
        from osworld_cube.computer import ComputerConfig

        computer = ComputerConfig(observe_after_action=False).make()
        mock_guest = _make_mock_guest()
        computer._guest = mock_guest

        result = computer.click(x=100, y=200)
        assert result == "Success"
        mock_guest.execute_action.assert_called_once()
        call_args = mock_guest.execute_action.call_args[0][0]
        assert call_args["action_type"] == "CLICK"
        assert call_args["parameters"]["x"] == 100
        assert call_args["parameters"]["y"] == 200

    def test_done_sets_is_done(self) -> None:
        from osworld_cube.computer import ComputerConfig

        computer = ComputerConfig(observe_after_action=False).make()
        assert computer._is_done is False
        computer.done()
        assert computer._is_done is True

    def test_fail_sets_is_done(self) -> None:
        from osworld_cube.computer import ComputerConfig

        computer = ComputerConfig(observe_after_action=False).make()
        computer.fail()
        assert computer._is_done is True

    def test_update_marks_and_run_pyautogui(self) -> None:
        from osworld_cube.computer import ComputerConfig

        computer = ComputerConfig(action_space="pyautogui", observe_after_action=False).make()
        mock_guest = _make_mock_guest()
        computer._guest = mock_guest

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

        computer = ComputerConfig(observe_after_action=False).make()
        mock_guest = _make_mock_guest()
        computer._guest = mock_guest

        result = computer.execute_action(Action(name="typing", arguments={"text": "hello"}))
        assert isinstance(result, Observation)
        mock_guest.execute_action.assert_called_once()

    def test_close_does_not_stop_vm(self) -> None:
        """ComputerBase.close() must NOT stop the VM -- caller owns VM lifecycle."""
        from osworld_cube.computer import ComputerConfig

        computer = ComputerConfig().make()
        mock_vm = MagicMock()
        computer._vm = mock_vm
        computer.close()
        mock_vm.stop.assert_not_called()


# ---------------------------------------------------------------------------
# OSWorldTask helpers
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


def _make_mock_vm(server_port: int = 15000, chromium_port: int = 19222, vlc_port: int = 18080) -> MagicMock:
    """Return a Mock that looks like LocalQEMUVM."""
    vm = MagicMock()
    vm.endpoint = f"http://localhost:{server_port}"
    vm.server_port = server_port
    vm.chromium_port = chromium_port
    vm.vlc_port = vlc_port
    return vm


# ---------------------------------------------------------------------------
# OSWorldTask
# ---------------------------------------------------------------------------


class TestOSWorldTask:
    def test_model_post_init_creates_tool_without_vm(self) -> None:
        from osworld_cube.computer import ComputerConfig
        from osworld_cube.task import OSWorldTask

        task = OSWorldTask(metadata=_make_task_metadata(), tool_config=ComputerConfig())
        assert task._computer is not None
        assert task._computer._vm is None
        assert task._vm is None

    def test_reset_with_vm_backend_launches_vm(self) -> None:
        from osworld_cube.computer import ComputerConfig
        from osworld_cube.task import OSWorldTask
        from osworld_cube.vm_backend import LocalQEMUVMBackend

        mock_vm = _make_mock_vm()
        mock_backend = MagicMock(spec=LocalQEMUVMBackend)
        mock_backend.launch.return_value = mock_vm

        with (
            patch(PATCH_GUEST_AGENT) as mock_ga_cls,
            patch(PATCH_SETUP_CTRL),
            patch(PATCH_SLEEP),
        ):
            mock_guest = _make_mock_guest()
            mock_ga_cls.return_value = mock_guest

            task = OSWorldTask(
                metadata=_make_task_metadata(),
                tool_config=ComputerConfig(),
                vm_backend=mock_backend,
            )
            obs, info = task.reset()

        mock_backend.launch.assert_called_once()
        assert task._vm is mock_vm
        assert isinstance(obs, Observation)

    def test_reset_returns_obs_and_info(self) -> None:
        from osworld_cube.computer import ComputerConfig
        from osworld_cube.task import OSWorldTask

        task = OSWorldTask(metadata=_make_task_metadata(), tool_config=ComputerConfig())
        mock_vm = _make_mock_vm()
        task._vm = mock_vm

        with (
            patch(PATCH_GUEST_AGENT) as mock_ga_cls,
            patch(PATCH_SETUP_CTRL),
            patch(PATCH_SLEEP),
        ):
            mock_guest = _make_mock_guest()
            mock_ga_cls.return_value = mock_guest
            task._computer.attach_vm(mock_vm)

            obs, info = task.reset()

        assert isinstance(obs, Observation)
        texts = [c.data for c in obs.contents if isinstance(c, TextContent)]
        assert any("Do something" in t for t in texts)
        assert info["task_id"] == "t1"
        assert info["task_domain"] == "os"

    def test_reset_resets_is_done(self) -> None:
        from osworld_cube.computer import ComputerConfig
        from osworld_cube.task import OSWorldTask

        task = OSWorldTask(metadata=_make_task_metadata(), tool_config=ComputerConfig())
        mock_vm = _make_mock_vm()
        task._vm = mock_vm

        with (
            patch(PATCH_GUEST_AGENT) as mock_ga_cls,
            patch(PATCH_SETUP_CTRL),
            patch(PATCH_SLEEP),
        ):
            mock_ga_cls.return_value = _make_mock_guest()
            task._computer.attach_vm(mock_vm)
            task._computer._is_done = True
            task.reset()
            assert task._computer._is_done is False

    def test_evaluate_calls_evaluator(self) -> None:
        from osworld_cube.computer import ComputerConfig
        from osworld_cube.task import OSWorldTask

        task = OSWorldTask(metadata=_make_task_metadata(), tool_config=ComputerConfig())
        mock_vm = _make_mock_vm()
        task._vm = mock_vm
        task._current_task_config = {"id": "t1", "evaluator": {"func": "check_file"}}
        task._computer._guest = _make_mock_guest()

        with patch(PATCH_EVALUATOR) as mock_eval_cls:
            mock_eval_cls.return_value.evaluate.return_value = 0.5
            reward, info = task.evaluate(Observation.from_text("state"))

        assert reward == 0.5
        assert "evaluator" in info

    def test_evaluate_no_evaluator_returns_zero(self) -> None:
        from cube.task import TaskMetadata

        from osworld_cube.computer import ComputerConfig
        from osworld_cube.task import OSWorldTask

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

        task = OSWorldTask(metadata=_make_task_metadata(), tool_config=ComputerConfig())
        assert task.finished(Observation()) is False
        task._computer._is_done = True
        assert task.finished(Observation()) is True

    def test_step_done_action_triggers_evaluate(self) -> None:
        """Full step loop: agent calls done() -> task.step() -> EnvironmentOutput.done is True."""
        from osworld_cube.computer import ComputerConfig
        from osworld_cube.task import OSWorldTask

        task = OSWorldTask(
            metadata=_make_task_metadata(),
            tool_config=ComputerConfig(observe_after_action=False),
        )
        mock_vm = _make_mock_vm()
        task._vm = mock_vm

        with (
            patch(PATCH_GUEST_AGENT) as mock_ga_cls,
            patch(PATCH_SETUP_CTRL),
            patch(PATCH_SLEEP),
            patch(PATCH_EVALUATOR) as mock_eval_cls,
        ):
            mock_ga_cls.return_value = _make_mock_guest()
            mock_eval_cls.return_value.evaluate.return_value = 1.0
            task._computer.attach_vm(mock_vm)
            task.reset()

            env_out = task.step(Action(name="done", arguments={}))

        assert env_out.done is True
        assert env_out.reward == 1.0

    def test_step_click_not_done(self) -> None:
        """A regular action does not set done."""
        from osworld_cube.computer import ComputerConfig
        from osworld_cube.task import OSWorldTask

        task = OSWorldTask(
            metadata=_make_task_metadata(),
            tool_config=ComputerConfig(observe_after_action=False),
        )
        mock_vm = _make_mock_vm()
        task._vm = mock_vm

        with (
            patch(PATCH_GUEST_AGENT) as mock_ga_cls,
            patch(PATCH_SETUP_CTRL),
            patch(PATCH_SLEEP),
        ):
            mock_ga_cls.return_value = _make_mock_guest()
            task._computer.attach_vm(mock_vm)
            task.reset()

            env_out = task.step(Action(name="click", arguments={"x": 10, "y": 20}))
        assert env_out.done is False

    def test_close_stops_vm(self) -> None:
        from osworld_cube.computer import ComputerConfig
        from osworld_cube.task import OSWorldTask

        task = OSWorldTask(metadata=_make_task_metadata(), tool_config=ComputerConfig())
        mock_vm = _make_mock_vm()
        task._vm = mock_vm
        task._computer._guest = _make_mock_guest()

        task.close()
        mock_vm.stop.assert_called_once()
        assert task._vm is None


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
        from osworld_cube.benchmark import OSWORLD_REPO_DIR, OSWorldBenchmark
        from osworld_cube.computer import ComputerConfig

        bench = OSWorldBenchmark(default_tool_config=ComputerConfig())
        task_data = {
            "id": "t",
            "config": [{"type": "setup", "parameters": {"settings_file": "configs/x.json"}}],
        }
        fixed = bench._fix_settings_paths(task_data)

        assert fixed["config"][0]["parameters"]["settings_file"] == str(OSWORLD_REPO_DIR / "configs/x.json")

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
