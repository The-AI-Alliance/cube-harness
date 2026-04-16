"""WAA task evaluator: ported from desktop_env.desktop_env.evaluate().

GuestAgentProxy mimics the DesktopEnv interface expected by getter functions.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable

from cube_computer_tool.guest_agent import GuestAgent

from waa_cube.vm_backend import getters, metrics

logger = logging.getLogger(__name__)

Metric = Callable[..., float]
Getter = Callable[..., Any]


class GuestAgentProxy:
    """Thin adapter exposing the same attributes WAA getter functions expect from DesktopEnv.

    Getter functions call ``env.controller.*``, ``env.cache_dir``,
    ``env.chromium_port``, ``env.vm_ip``, etc.  This proxy provides exactly
    those attributes backed by a :class:`GuestAgent` instance.
    """

    def __init__(
        self,
        guest: GuestAgent,
        cache_dir: str,
        chromium_port: int,
        vlc_port: int,
        server_port: int,
        vm_ip: str = "localhost",
    ) -> None:
        # getters call env.controller.* — point directly at the GuestAgent
        self.controller = guest
        self.cache_dir = cache_dir
        self.chromium_port = chromium_port
        self.vlc_port = vlc_port
        self.server_port = server_port
        self.vm_ip = vm_ip
        self.current_use_proxy = False  # chrome getters check this for proxy routing

    @property
    def vm_platform(self) -> str:
        return self.controller.get_vm_platform()


class Evaluator:
    """Runs WAA task evaluation using ported getter + metric functions.

    Mirrors ``DesktopEnv.evaluate()`` but uses :class:`GuestAgentProxy` instead
    of the full DesktopEnv instance, and accepts the action history as a parameter.
    """

    def __init__(
        self,
        guest: GuestAgent,
        cache_dir_base: Path,
        chromium_port: int,
        vlc_port: int,
        server_port: int,
        vm_ip: str = "localhost",
    ) -> None:
        self._guest = guest
        self._cache_dir_base = cache_dir_base
        self._chromium_port = chromium_port
        self._vlc_port = vlc_port
        self._server_port = server_port
        self._vm_ip = vm_ip

    def evaluate(self, task_config: dict[str, Any], action_history: list[Any]) -> float:
        """Evaluate the task and return a reward in [0.0, 1.0].

        Parameters
        ----------
        task_config : dict
            Full task configuration dict (as loaded from WAA JSON files).
        action_history : list
            Sequence of actions taken by the agent during the episode.
        """
        evaluator_cfg: dict[str, Any] = task_config["evaluator"]
        task_id: str = task_config.get("id", "unknown")

        task_cache_dir = str(self._cache_dir_base / task_id)
        os.makedirs(task_cache_dir, exist_ok=True)

        env_proxy = GuestAgentProxy(
            guest=self._guest,
            cache_dir=task_cache_dir,
            chromium_port=self._chromium_port,
            vlc_port=self._vlc_port,
            server_port=self._server_port,
            vm_ip=self._vm_ip,
        )

        # Postconfig: run any cleanup setup steps before evaluation
        postconfig = evaluator_cfg.get("postconfig", [])
        if postconfig:
            from waa_cube.vm_backend.setup_controller import SetupController

            setup_ctrl = SetupController(
                guest=self._guest,
                chromium_port=self._chromium_port,
                vlc_port=self._vlc_port,
                cache_dir=task_cache_dir,
                screen_width=1920,
                screen_height=1080,
            )
            setup_ctrl.setup(postconfig)

        # Infeasible tasks: reward 1 only if agent called FAIL
        if evaluator_cfg.get("func") == "infeasible":
            if action_history and action_history[-1] == "FAIL":
                return 1.0
            return 0.0

        # Non-infeasible tasks: FAIL means 0
        if action_history and action_history[-1] == "FAIL":
            return 0.0

        is_list = isinstance(evaluator_cfg.get("func"), list)
        if is_list:
            return self._evaluate_multiple(evaluator_cfg, env_proxy)
        return self._evaluate_single(evaluator_cfg, env_proxy)

    def _evaluate_single(self, evaluator_cfg: dict[str, Any], env_proxy: GuestAgentProxy) -> float:
        metric_fn: Metric = getattr(metrics, evaluator_cfg["func"])
        result_getter_cfg = evaluator_cfg.get("result")
        expected_getter_cfg = evaluator_cfg.get("expected")
        options: dict = evaluator_cfg.get("options") or {}

        try:
            result_state = self._call_getter(result_getter_cfg, env_proxy) if result_getter_cfg else None
        except FileNotFoundError:
            logger.error("File not found during evaluation")
            return 0.0
        except Exception as exc:
            logger.error("Unexpected error during result getter: %s", exc)
            return 0.0

        if expected_getter_cfg:
            expected_state = self._call_getter(expected_getter_cfg, env_proxy)
            return float(metric_fn(result_state, expected_state, **options))
        return float(metric_fn(result_state, **options))

    def _evaluate_multiple(self, evaluator_cfg: dict[str, Any], env_proxy: GuestAgentProxy) -> float:
        conj: str = evaluator_cfg.get("conj", "and")
        func_names: list[str] = evaluator_cfg["func"]
        result_cfgs: list = evaluator_cfg.get("result") or [None] * len(func_names)
        expected_cfgs: list = evaluator_cfg.get("expected") or [None] * len(func_names)
        options_list: list[dict] = (
            [o or {} for o in evaluator_cfg["options"]]
            if isinstance(evaluator_cfg.get("options"), list)
            else [{}] * len(func_names)
        )

        results: list[float] = []
        for func_name, result_cfg, expected_cfg, opts in zip(func_names, result_cfgs, expected_cfgs, options_list):
            metric_fn: Metric = getattr(metrics, func_name)

            try:
                result_state = self._call_getter(result_cfg, env_proxy) if result_cfg else None
            except FileNotFoundError:
                logger.error("File not found for metric %s", func_name)
                if conj == "and":
                    return 0.0
                results.append(0.0)
                continue
            except Exception as exc:
                logger.error("Unexpected error for metric %s: %s", func_name, exc)
                if conj == "and":
                    return 0.0
                results.append(0.0)
                continue

            if expected_cfg:
                expected_state = self._call_getter(expected_cfg, env_proxy)
                score = float(metric_fn(result_state, expected_state, **opts))
            else:
                score = float(metric_fn(result_state, **opts))

            if conj == "and" and score == 0.0:
                return 0.0
            if conj == "or" and score == 1.0:
                return 1.0
            results.append(score)

        if not results:
            return 0.0
        return sum(results) / len(results) if conj == "and" else max(results)

    @staticmethod
    def _call_getter(cfg: dict[str, Any], env_proxy: GuestAgentProxy) -> Any:
        getter_fn: Getter = getattr(getters, "get_{:}".format(cfg["type"]))
        return getter_fn(env_proxy, cfg)
