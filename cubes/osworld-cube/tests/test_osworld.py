"""Tests for osworld_cube — verifies compliance with the CUBE protocol ABCs."""

from __future__ import annotations

import io
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from cube.core import Action, Observation, TextContent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_screenshot_bytes(w: int = 100, h: int = 100) -> bytes:
    """Return a minimal PNG screenshot as bytes."""
    img = Image.new("RGB", (w, h), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_mock_vm(endpoint: str = "http://localhost:9999") -> MagicMock:
    """Return a mock VM object compatible with cube.vm.VM interface."""
    vm = MagicMock()
    vm.endpoint = endpoint
    return vm


def _make_cct_config(
    require_a11y_tree: bool = False,
    require_terminal: bool = False,
    observe_after_action: bool = False,
) -> MagicMock:
    """Return a mock _CCTComputerConfig."""
    from cube.vm import VMConfig

    cfg = MagicMock()
    cfg.require_a11y_tree = require_a11y_tree
    cfg.require_terminal = require_terminal
    cfg.observe_after_action = observe_after_action
    cfg.vm_config = VMConfig()
    return cfg


def _make_computer(
    require_a11y_tree: bool = False,
    require_terminal: bool = False,
    observe_after_action: bool = False,
    os_type: str = "Ubuntu",
) -> tuple:
    """Directly construct a ComputerBase with mock dependencies.

    Returns (computer, mock_vm). Does NOT call ComputerConfig.make() so no
    Docker/download patching is needed.
    """
    from osworld_cube.computer import ComputerBase

    mock_vm = _make_mock_vm()
    cct_config = _make_cct_config(
        require_a11y_tree=require_a11y_tree,
        require_terminal=require_terminal,
        observe_after_action=observe_after_action,
    )
    computer = ComputerBase(config=cct_config, vm=mock_vm, os_type=os_type)
    return computer, mock_vm


def _make_screenshot_response() -> MagicMock:
    """Return a mock requests.Response carrying a PNG screenshot."""
    resp = MagicMock()
    resp.content = _make_screenshot_bytes()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    return resp


def _make_post_response(returncode: int = 0) -> MagicMock:
    """Return a mock requests.Response for a POST /execute call."""
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"returncode": returncode}
    return resp


# Patch targets
PATCH_LAUNCH = "cube_computer_tool.backends.local_qemu.LocalQEMUVMBackend.launch"
PATCH_SLEEP = "cube_computer_tool.computer.time.sleep"
PATCH_REQUESTS_GET = "cube_computer_tool.computer.requests.get"
PATCH_REQUESTS_POST = "cube_computer_tool.computer.requests.post"
PATCH_GET_VM = "osworld_cube.vm_utils.get_osworld_vm_image"


@contextmanager
def _task_ctx(metadata=None, require_a11y_tree: bool = False, observe_after_action: bool = False):
    """Context manager: create an OSWorldTask with all VM/HTTP calls mocked.

    Yields (task, mock_vm). The mock_vm.endpoint is set to a test URL.
    PATCH_REQUESTS_GET and PATCH_SLEEP are active for the entire context so
    callers can call task.reset() inside the context without extra patching.
    """
    from osworld_cube.computer import ComputerConfig
    from osworld_cube.task import OSWorldTask

    if metadata is None:
        metadata = _make_task_metadata()

    mock_vm = _make_mock_vm()

    with (
        patch(PATCH_LAUNCH, return_value=mock_vm),
        patch(PATCH_GET_VM, return_value="/fake/Ubuntu.qcow2"),
        patch(PATCH_SLEEP),
        patch(PATCH_REQUESTS_GET, return_value=_make_screenshot_response()),
    ):
        task = OSWorldTask(
            metadata=metadata,
            tool_config=ComputerConfig(
                require_a11y_tree=require_a11y_tree,
                observe_after_action=observe_after_action,
            ),
        )
        yield task, mock_vm


# ---------------------------------------------------------------------------
# TaskMetadata helper
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


# ---------------------------------------------------------------------------
# ComputerConfig
# ---------------------------------------------------------------------------


class TestComputerConfig:
    def test_defaults(self):
        from osworld_cube.computer import ComputerConfig

        cfg = ComputerConfig()
        assert cfg.os_type == "Ubuntu"
        assert cfg.headless is True
        assert cfg.require_a11y_tree is True
        assert cfg.observe_after_action is True
        assert cfg.path_to_vm is None

    def test_custom_values(self):
        from osworld_cube.computer import ComputerConfig

        cfg = ComputerConfig(os_type="Windows", headless=False, require_a11y_tree=False)
        assert cfg.os_type == "Windows"
        assert cfg.headless is False
        assert cfg.require_a11y_tree is False

    def test_make_launches_vm_and_returns_computer_base(self):
        from osworld_cube.computer import ComputerBase, ComputerConfig

        mock_vm = _make_mock_vm()
        with patch(PATCH_LAUNCH, return_value=mock_vm) as mock_launch, patch(
            PATCH_GET_VM, return_value="/fake/Ubuntu.qcow2"
        ):
            computer = ComputerConfig().make()

        assert mock_launch.called
        assert isinstance(computer, ComputerBase)
        assert computer.config.os_type == "Ubuntu"

    def test_path_to_vm_skips_download(self):
        """path_to_vm=... bypasses get_osworld_vm_image()."""
        from osworld_cube.computer import ComputerConfig

        mock_vm = _make_mock_vm()
        with patch(PATCH_LAUNCH, return_value=mock_vm), patch(
            PATCH_GET_VM, side_effect=AssertionError("should not be called")
        ):
            # Should not raise — get_osworld_vm_image is not called
            ComputerConfig(path_to_vm="/already/there.qcow2").make()


# ---------------------------------------------------------------------------
# ComputerBase
# ---------------------------------------------------------------------------


class TestComputer:
    def test_action_set_contains_expected_actions(self):
        computer, _ = _make_computer()
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

    def test_config_returns_os_type(self):
        computer, _ = _make_computer(os_type="Ubuntu")
        assert computer.config.os_type == "Ubuntu"

    def test_update_marks_stores_marks(self):
        computer, _ = _make_computer()
        marks = [[10, 20, 30, 40], [50, 60, 10, 10]]
        computer.update_marks(marks)
        assert computer._marks == marks

    def test_done_sets_is_done(self):
        computer, _ = _make_computer()
        assert computer._is_done is False
        computer.done()
        assert computer._is_done is True

    def test_fail_sets_is_done(self):
        computer, _ = _make_computer()
        computer.fail()
        assert computer._is_done is True

    def test_evaluate_before_setup_returns_zero(self):
        computer, _ = _make_computer()
        # _task_config is None before setup_task()
        assert computer.evaluate_task() == 0.0

    def test_evaluate_no_evaluator_returns_zero(self):
        computer, _ = _make_computer()
        computer._task_config = {"id": "t1", "evaluator": {}}
        assert computer.evaluate_task() == 0.0

    def test_evaluate_unsupported_func_returns_zero(self):
        computer, _ = _make_computer()
        computer._task_config = {
            "id": "t1",
            "evaluator": {"func": "check_url", "expected": {}},
        }
        assert computer.evaluate_task() == 0.0

    def test_evaluate_check_include_exclude_pass(self):
        computer, _ = _make_computer()
        computer._task_config = {
            "id": "t1",
            "evaluator": {
                "func": "check_include_exclude",
                "result": {"type": "vm_command_line", "command": "cat /tmp/out.txt"},
                "expected": {"rules": {"include": ["hello"], "exclude": ["error"]}},
            },
        }
        with patch.object(computer, "run_shell_command", return_value="hello world"):
            assert computer.evaluate_task() == 1.0

    def test_evaluate_check_include_exclude_fail_missing_include(self):
        computer, _ = _make_computer()
        computer._task_config = {
            "id": "t1",
            "evaluator": {
                "func": "check_include_exclude",
                "result": {"type": "vm_command_line", "command": "ls"},
                "expected": {"rules": {"include": ["missing_string"], "exclude": []}},
            },
        }
        with patch.object(computer, "run_shell_command", return_value="something else"):
            assert computer.evaluate_task() == 0.0

    def test_evaluate_check_include_exclude_fail_excluded_present(self):
        computer, _ = _make_computer()
        computer._task_config = {
            "id": "t1",
            "evaluator": {
                "func": "check_include_exclude",
                "result": {"type": "vm_command_line", "command": "ls"},
                "expected": {"rules": {"include": [], "exclude": ["forbidden"]}},
            },
        }
        with patch.object(computer, "run_shell_command", return_value="forbidden stuff"):
            assert computer.evaluate_task() == 0.0

    def test_setup_task_returns_observation(self):
        computer, mock_vm = _make_computer(require_a11y_tree=False, observe_after_action=False)

        with (
            patch(PATCH_REQUESTS_GET, return_value=_make_screenshot_response()),
            patch(PATCH_SLEEP),
        ):
            obs = computer.setup_task(
                {"id": "t1", "instruction": "test", "config": [], "snapshot": "init_state"}
            )

        assert isinstance(obs, Observation)
        mock_vm.restore_snapshot.assert_called_once_with("init_state")

    def test_setup_task_waits_60_seconds(self):
        computer, _ = _make_computer(require_a11y_tree=False, observe_after_action=False)

        with (
            patch(PATCH_REQUESTS_GET, return_value=_make_screenshot_response()),
            patch(PATCH_SLEEP) as mock_sleep,
        ):
            computer.setup_task({"id": "t1", "config": [], "snapshot": "init_state"})

        mock_sleep.assert_called_once_with(60)

    def test_click_dispatches_to_http(self):
        computer, _ = _make_computer(observe_after_action=False)

        with patch(PATCH_REQUESTS_POST, return_value=_make_post_response()) as mock_post:
            result = computer.click(x=100, y=200)

        assert result == "Success"
        assert mock_post.called
        # Verify the POST was to /execute on the endpoint
        call_url = mock_post.call_args[0][0]
        assert "/execute" in call_url

    def test_execute_action_dispatches_via_cube(self):
        """cube.tool.Tool.execute_action routes by action name to the correct @tool_action."""
        computer, _ = _make_computer(observe_after_action=False)

        with patch(PATCH_REQUESTS_POST, return_value=_make_post_response()):
            result = computer.execute_action(Action(name="typing", arguments={"text": "hello"}))

        assert isinstance(result, Observation)

    def test_close_stops_vm(self):
        computer, mock_vm = _make_computer()
        computer.close()
        mock_vm.stop.assert_called_once()


# ---------------------------------------------------------------------------
# OSWorldTask
# ---------------------------------------------------------------------------


class TestOSWorldTask:
    def test_reset_returns_obs_and_info(self):
        with _task_ctx() as (task, _):
            obs, info = task.reset()

        assert isinstance(obs, Observation)
        texts = [c.data for c in obs.contents if isinstance(c, TextContent)]
        assert any("Do something" in t for t in texts)
        assert info["task_id"] == "t1"
        assert info["task_domain"] == "os"

    def test_reset_resets_is_done(self):
        with _task_ctx() as (task, _):
            task._computer._is_done = True
            task.reset()
            assert task._computer._is_done is False

    def test_evaluate_returns_reward_and_info(self):
        with _task_ctx() as (task, _):
            with patch.object(task._computer, "evaluate_task", return_value=0.5):
                reward, info = task.evaluate(Observation())

        assert reward == 0.5
        assert "evaluator" in info

    def test_evaluate_no_evaluator_returns_zero(self):
        from cube.task import TaskMetadata

        meta = TaskMetadata(id="no-eval", extra_info={})
        with _task_ctx(metadata=meta) as (task, _):
            reward, info = task.evaluate(Observation())

        assert reward == 0.0
        assert info.get("error") == "no_evaluator"

    def test_finished_reflects_computer_is_done(self):
        with _task_ctx() as (task, _):
            assert task.finished(Observation()) is False
            task._computer._is_done = True
            assert task.finished(Observation()) is True

    def test_step_done_action_triggers_evaluate(self):
        """Full step loop: agent calls done() → task sets done=True → evaluate()."""
        with _task_ctx() as (task, _):
            task.reset()
            with patch.object(task._computer, "evaluate_task", return_value=1.0):
                env_out = task.step(Action(name="done", arguments={}))

        assert env_out.done is True
        assert env_out.reward == 1.0

    def test_step_click_not_done(self):
        """A regular action does not set done."""
        with _task_ctx() as (task, _):
            task.reset()
            with patch(PATCH_REQUESTS_POST, return_value=_make_post_response()):
                env_out = task.step(Action(name="click", arguments={"x": 10, "y": 20}))

        assert env_out.done is False

    def test_close_calls_vm_stop(self):
        with _task_ctx() as (task, mock_vm):
            task.reset()
            task.close()

        mock_vm.stop.assert_called_once()


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
    def test_benchmark_metadata(self):
        from osworld_cube.benchmark import OSWorldBenchmark

        assert OSWorldBenchmark.benchmark_metadata.name == "osworld"
        assert OSWorldBenchmark.task_config_class.__name__ == "OSWorldTaskConfig"

    def test_load_all_tasks_from_repo(self):
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

    def test_get_task_configs_carries_metadata(self):
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

    def test_task_config_make_produces_osworld_task(self):
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

            with (
                patch(PATCH_LAUNCH, return_value=_make_mock_vm()),
                patch(PATCH_GET_VM, return_value="/fake/Ubuntu.qcow2"),
            ):
                task = cfg.make()

        assert isinstance(task, OSWorldTask)
        assert task.metadata.id == cfg.task_id

    def test_load_from_flat_json_file(self):
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

    def test_fix_settings_paths(self):
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
