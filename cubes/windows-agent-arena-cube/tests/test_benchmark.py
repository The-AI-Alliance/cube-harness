"""Tests for waa_cube — verifies compliance with the CUBE protocol ABCs."""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from cube.core import Action, Observation, TextContent

# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------

PATCH_GUEST_AGENT = "cube_computer_tool.computer.GuestAgent"
PATCH_EVALUATOR = "waa_cube.task.Evaluator"
PATCH_SETUP_CTRL = "waa_cube.task.SetupController"
PATCH_SLEEP = "waa_cube.task.time.sleep"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_screenshot_bytes(w: int = 100, h: int = 100) -> bytes:
    img = Image.new("RGB", (w, h), color=(64, 64, 64))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_mock_guest(screenshot: bytes | None = None, axtree: str = "<root/>") -> MagicMock:
    guest = MagicMock()
    guest.get_screenshot.return_value = screenshot or _make_screenshot_bytes()
    guest.get_accessibility_tree.return_value = axtree
    guest.get_terminal_output.return_value = ""
    guest.execute_action.return_value = None
    guest.execute_python_command.return_value = {"returncode": 0, "output": ""}
    return guest


def _make_mock_vm(
    server_port: int = 15000,
    chromium_port: int = 19222,
    vlc_port: int = 18080,
) -> MagicMock:
    vm = MagicMock()
    vm.endpoint = f"http://localhost:{server_port}"
    vm.server_port = server_port
    vm.chromium_port = chromium_port
    vm.vlc_port = vlc_port
    return vm


def _make_task_metadata(
    task_id: str = "t1",
    instruction: str = "Do something",
    snapshot: str = "vscode",
) -> "TaskMetadata":  # type: ignore[name-defined]
    from cube.task import TaskMetadata

    return TaskMetadata(
        id=task_id,
        abstract_description=instruction,
        extra_info={
            "domain": "vscode",
            "snapshot": snapshot,
            "config": [],
            "evaluator": {"func": "check_json_settings", "expected": {}},
            "related_apps": ["vscode"],
        },
    )


def _make_waa_eval_dir(tmpdir: Path) -> Path:
    """Create a minimal fake evaluation_examples_windows/ with 2 tasks."""
    eval_dir = tmpdir / "evaluation_examples_windows"
    (eval_dir / "examples" / "vscode").mkdir(parents=True)
    (eval_dir / "examples" / "chrome").mkdir(parents=True)

    test_set = {"vscode": ["vs-1"], "chrome": ["ch-1"]}
    (eval_dir / "test_all.json").write_text(json.dumps(test_set))

    (eval_dir / "examples" / "vscode" / "vs-1.json").write_text(
        json.dumps(
            {
                "id": "vs-1",
                "instruction": "Open VS Code",
                "snapshot": "vscode",
                "config": [{"type": "launch", "parameters": {"command": ["code"]}}],
                "evaluator": {"func": "check_json_settings", "expected": {}},
                "related_apps": ["vscode"],
            }
        )
    )
    (eval_dir / "examples" / "chrome" / "ch-1.json").write_text(
        json.dumps(
            {
                "id": "ch-1",
                "instruction": "Open Chrome",
                "snapshot": "chrome",
                "config": [],
                "evaluator": {"func": "is_expected_tabs", "expected": {}},
                "related_apps": ["chrome"],
            }
        )
    )
    return eval_dir


# ---------------------------------------------------------------------------
# ComputerConfig
# ---------------------------------------------------------------------------


class TestComputerConfig:
    def test_defaults(self) -> None:
        from waa_cube.computer import ComputerConfig

        cfg = ComputerConfig()
        assert cfg.require_a11y_tree is True

    def test_cache_dir_is_waa(self) -> None:
        from waa_cube.computer import ComputerConfig

        cfg = ComputerConfig()
        assert "waa" in cfg.cache_dir.lower()

    def test_make_without_vm_succeeds(self) -> None:
        from waa_cube.computer import ComputerConfig

        computer = ComputerConfig().make()
        assert computer._vm is None
        assert computer._guest is None


# ---------------------------------------------------------------------------
# WAATask
# ---------------------------------------------------------------------------


class TestWAATask:
    def test_model_post_init_creates_tool_without_vm(self) -> None:
        from waa_cube.computer import ComputerConfig
        from waa_cube.task import WAATask

        task = WAATask(metadata=_make_task_metadata(), tool_config=ComputerConfig())
        assert task._computer is not None
        assert task._computer._vm is None
        assert task._vm is None

    def test_os_type_is_always_windows(self) -> None:
        from waa_cube.computer import ComputerConfig
        from waa_cube.task import WAATask

        task = WAATask(metadata=_make_task_metadata(), tool_config=ComputerConfig())
        assert task._os_type() == "windows"

    def test_reset_with_vm_backend_launches_vm(self) -> None:
        from cube.vm import VMBackend

        from waa_cube.computer import ComputerConfig
        from waa_cube.task import WAATask

        mock_vm = _make_mock_vm()
        mock_backend = MagicMock(spec=VMBackend)
        mock_backend.launch.return_value = mock_vm

        with (
            patch(PATCH_GUEST_AGENT) as mock_ga_cls,
            patch(PATCH_SETUP_CTRL),
            patch(PATCH_SLEEP),
        ):
            mock_ga_cls.return_value = _make_mock_guest()
            task = WAATask(
                metadata=_make_task_metadata(),
                tool_config=ComputerConfig(),
                vm_backend=mock_backend,
            )
            obs, info = task.reset()

        mock_backend.launch.assert_called_once()
        assert task._vm is mock_vm
        assert isinstance(obs, Observation)

    def test_reset_returns_goal_in_obs(self) -> None:
        from waa_cube.computer import ComputerConfig
        from waa_cube.task import WAATask

        task = WAATask(metadata=_make_task_metadata(), tool_config=ComputerConfig())
        mock_vm = _make_mock_vm()
        task._vm = mock_vm

        with (
            patch(PATCH_GUEST_AGENT) as mock_ga_cls,
            patch(PATCH_SETUP_CTRL),
            patch(PATCH_SLEEP),
        ):
            mock_ga_cls.return_value = _make_mock_guest()
            task._computer.attach_vm(mock_vm)
            obs, info = task.reset()

        texts = [c.data for c in obs.contents if isinstance(c, TextContent)]
        assert any("Do something" in t for t in texts)
        assert info["task_id"] == "t1"
        assert info["task_snapshot"] == "vscode"

    def test_evaluate_calls_evaluator(self) -> None:
        from waa_cube.computer import ComputerConfig
        from waa_cube.task import WAATask

        task = WAATask(metadata=_make_task_metadata(), tool_config=ComputerConfig())
        task._vm = _make_mock_vm()
        task._computer._guest = _make_mock_guest()

        with patch(PATCH_EVALUATOR) as mock_eval_cls:
            mock_eval_cls.return_value.evaluate.return_value = 1.0
            reward, info = task.evaluate(Observation.from_text("state"))

        assert reward == 1.0
        assert "evaluator" in info

    def test_evaluate_no_evaluator_returns_zero(self) -> None:
        from cube.task import TaskMetadata

        from waa_cube.computer import ComputerConfig
        from waa_cube.task import WAATask

        task = WAATask(
            metadata=TaskMetadata(id="no-eval", extra_info={}),
            tool_config=ComputerConfig(),
        )
        reward, info = task.evaluate(Observation())
        assert reward == 0.0
        assert info.get("error") == "no_evaluator"

    def test_finished_reflects_computer_is_done(self) -> None:
        from waa_cube.computer import ComputerConfig
        from waa_cube.task import WAATask

        task = WAATask(metadata=_make_task_metadata(), tool_config=ComputerConfig())
        assert task.finished(Observation()) is False
        task._computer._is_done = True
        assert task.finished(Observation()) is True

    def test_close_stops_vm(self) -> None:
        from waa_cube.computer import ComputerConfig
        from waa_cube.task import WAATask

        task = WAATask(metadata=_make_task_metadata(), tool_config=ComputerConfig())
        mock_vm = _make_mock_vm()
        task._vm = mock_vm
        task._computer._guest = _make_mock_guest()

        task.close()
        mock_vm.stop.assert_called_once()
        assert task._vm is None

    def test_step_done_triggers_evaluate(self) -> None:
        from waa_cube.computer import ComputerConfig
        from waa_cube.task import WAATask

        task = WAATask(
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

    def test_restore_snapshot_called_on_reset(self) -> None:
        """VM.restore_snapshot() is called with the task's snapshot name."""
        from waa_cube.computer import ComputerConfig
        from waa_cube.task import WAATask

        task = WAATask(
            metadata=_make_task_metadata(snapshot="vscode"),
            tool_config=ComputerConfig(),
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

        mock_vm.restore_snapshot.assert_called_once_with("vscode")


# ---------------------------------------------------------------------------
# WAABenchmark
# ---------------------------------------------------------------------------


class TestWAABenchmark:
    def test_benchmark_metadata(self) -> None:
        from waa_cube.benchmark import WAABenchmark

        assert WAABenchmark.benchmark_metadata.name == "waa"
        assert WAABenchmark.benchmark_metadata.num_tasks == 154
        assert "windows" in WAABenchmark.benchmark_metadata.tags
        assert WAABenchmark.task_config_class.__name__ == "WAATaskConfig"

    def test_load_tasks_from_eval_dir(self) -> None:
        from waa_cube.benchmark import WAABenchmark
        from waa_cube.computer import ComputerConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            eval_dir = _make_waa_eval_dir(Path(tmpdir))
            bench = WAABenchmark(
                eval_examples_dir=str(eval_dir),
                default_tool_config=ComputerConfig(),
            )
            bench.setup()

            assert len(bench.task_metadata) == 2
            assert "vs-1" in bench.task_metadata
            assert "ch-1" in bench.task_metadata

    def test_task_metadata_extra_info(self) -> None:
        from waa_cube.benchmark import WAABenchmark
        from waa_cube.computer import ComputerConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            eval_dir = _make_waa_eval_dir(Path(tmpdir))
            bench = WAABenchmark(eval_examples_dir=str(eval_dir), default_tool_config=ComputerConfig())
            bench.setup()

            meta = bench.task_metadata["vs-1"]
            assert meta.extra_info["domain"] == "vscode"
            assert meta.extra_info["snapshot"] == "vscode"
            assert meta.abstract_description == "Open VS Code"

    def test_subset_from_glob_filters_domain(self) -> None:
        from waa_cube.benchmark import WAABenchmark
        from waa_cube.computer import ComputerConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            eval_dir = _make_waa_eval_dir(Path(tmpdir))
            bench = WAABenchmark(eval_examples_dir=str(eval_dir), default_tool_config=ComputerConfig())
            bench.setup()

            vscode_bench = bench.subset_from_glob("extra_info.domain", "vscode")
            assert len(vscode_bench.task_metadata) == 1
            assert "vs-1" in vscode_bench.task_metadata

    def test_get_task_configs_yields_waa_task_config(self) -> None:
        from waa_cube.benchmark import WAABenchmark, WAATaskConfig
        from waa_cube.computer import ComputerConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            eval_dir = _make_waa_eval_dir(Path(tmpdir))
            bench = WAABenchmark(eval_examples_dir=str(eval_dir), default_tool_config=ComputerConfig())
            bench.setup()

            configs = list(bench.get_task_configs())
            assert len(configs) == 2
            for cfg in configs:
                assert isinstance(cfg, WAATaskConfig)
                assert cfg.task_id in bench.task_metadata

    def test_task_config_make_produces_waa_task(self) -> None:
        from waa_cube.benchmark import WAABenchmark
        from waa_cube.computer import ComputerConfig
        from waa_cube.task import WAATask

        with tempfile.TemporaryDirectory() as tmpdir:
            eval_dir = _make_waa_eval_dir(Path(tmpdir))
            bench = WAABenchmark(eval_examples_dir=str(eval_dir), default_tool_config=ComputerConfig())
            bench.setup()

            cfg = next(bench.get_task_configs())
            task = cfg.make()

            assert isinstance(task, WAATask)
            assert task.metadata.id == cfg.task_id

    def test_vm_backend_injected_into_configs(self) -> None:
        from waa_cube.benchmark import WAABenchmark
        from waa_cube.computer import ComputerConfig
        from waa_cube.vm_backend.backend import WAADockerVMBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            eval_dir = _make_waa_eval_dir(Path(tmpdir))
            mock_backend = MagicMock(spec=WAADockerVMBackend)
            bench = WAABenchmark(
                eval_examples_dir=str(eval_dir),
                default_tool_config=ComputerConfig(),
                vm_backend=mock_backend,
            )
            bench.setup()

            for cfg in bench.get_task_configs():
                assert cfg.vm_backend is mock_backend

    def test_missing_eval_dir_raises(self) -> None:
        import os

        import pytest

        from waa_cube.benchmark import WAABenchmark, WAA_EVAL_EXAMPLES_ENV
        from waa_cube.computer import ComputerConfig

        # Ensure env var is not set
        env_backup = os.environ.pop(WAA_EVAL_EXAMPLES_ENV, None)
        try:
            bench = WAABenchmark(default_tool_config=ComputerConfig())
            with pytest.raises((ValueError, FileNotFoundError)):
                bench.setup()
        finally:
            if env_backup is not None:
                os.environ[WAA_EVAL_EXAMPLES_ENV] = env_backup

    def test_missing_task_file_logged_not_raised(self) -> None:
        """Missing individual task JSON files are logged but don't abort loading."""
        from waa_cube.benchmark import WAABenchmark
        from waa_cube.computer import ComputerConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            eval_dir = Path(tmpdir) / "evaluation_examples_windows"
            (eval_dir / "examples" / "vscode").mkdir(parents=True)
            (eval_dir / "test_all.json").write_text(
                json.dumps({"vscode": ["missing-task", "vs-1"]})
            )
            (eval_dir / "examples" / "vscode" / "vs-1.json").write_text(
                json.dumps(
                    {
                        "id": "vs-1",
                        "instruction": "Test",
                        "snapshot": "vscode",
                        "config": [],
                        "evaluator": {},
                        "related_apps": [],
                    }
                )
            )

            bench = WAABenchmark(eval_examples_dir=str(eval_dir), default_tool_config=ComputerConfig())
            bench.setup()

            # Only the task with a real file should be loaded
            assert len(bench.task_metadata) == 1
            assert "vs-1" in bench.task_metadata

    def test_close_does_not_raise(self) -> None:
        from waa_cube.benchmark import WAABenchmark
        from waa_cube.computer import ComputerConfig

        WAABenchmark(default_tool_config=ComputerConfig()).close()
