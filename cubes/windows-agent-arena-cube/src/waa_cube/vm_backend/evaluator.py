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
from waa_cube.vm_backend.setup_controller import SetupController

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
        self._cached_platform: str | None = None

    @property
    def vm_platform(self) -> str:
        """Return the VM's platform string, cached on first successful probe.

        Many getters call ``env.vm_platform`` to branch on OS; without caching
        every call re-probes via ``execute_python_command``, multiplying the
        chance one of them hits a transient guest-agent failure and returns
        ``""`` — which downstream gets misread as "Unsupported operating
        system". Cache the value once we have it and raise on probe failure
        so the harness surfaces a real error instead of silent reward=0.
        """
        if self._cached_platform is not None:
            return self._cached_platform
        platform_str = self.controller.get_vm_platform()
        if not platform_str:
            raise RuntimeError(
                "Failed to probe VM platform after retries — guest agent not "
                "responding. The VM tunnel may have collapsed mid-evaluation."
            )
        self._cached_platform = platform_str
        return platform_str


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

    def evaluate(self, task_config: dict[str, Any], action_history: list[Any]) -> tuple[float, dict[str, Any]]:
        """Evaluate the task and return ``(reward, info)``.

        ``reward`` ∈ [0.0, 1.0]. ``info`` carries diagnostic context that the
        harness records into ``trajectory.reward_info`` — in particular,
        ``evaluation_error`` is populated when a getter or metric raised, so
        a buggy evaluator surfaces as `error` data instead of being
        indistinguishable from a real agent failure (reward=0).

        Keys ``info`` may contain:
            - evaluator_func: the evaluator name(s) that ran
            - evaluation_error: {phase, func, type, message} on getter/metric failure
            - file_not_found: bool — distinguishes "missing eval artifact" from "code bug"
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
            reward = 1.0 if action_history and action_history[-1] == "FAIL" else 0.0
            return reward, {"evaluator_func": "infeasible"}

        # Non-infeasible tasks: FAIL means 0
        if action_history and action_history[-1] == "FAIL":
            return 0.0, {"evaluator_func": evaluator_cfg.get("func"), "agent_called_fail": True}

        if isinstance(evaluator_cfg.get("func"), list):
            return self._evaluate_multiple(evaluator_cfg, env_proxy)
        return self._evaluate_single(evaluator_cfg, env_proxy)

    def _evaluate_single(self, evaluator_cfg: dict[str, Any], env_proxy: GuestAgentProxy) -> tuple[float, dict[str, Any]]:
        func_name = evaluator_cfg["func"]
        info: dict[str, Any] = {"evaluator_func": func_name}
        try:
            metric_fn: Metric = getattr(metrics, func_name)
        except AttributeError as exc:
            return 0.0, {**info, "evaluation_error": {"phase": "metric_lookup", "func": func_name, "type": "AttributeError", "message": str(exc)}}
        result_getter_cfg = evaluator_cfg.get("result")
        expected_getter_cfg = evaluator_cfg.get("expected")
        options: dict = evaluator_cfg.get("options") or {}

        try:
            result_state = self._call_getter(result_getter_cfg, env_proxy) if result_getter_cfg else None
        except FileNotFoundError as exc:
            logger.error("File not found during evaluation: %s", exc)
            return 0.0, {**info, "file_not_found": True, "evaluation_error": {"phase": "result_getter", "func": func_name, "type": "FileNotFoundError", "message": str(exc)}}
        except Exception as exc:
            logger.exception("Unexpected error during result getter for %s", func_name)
            return 0.0, {**info, "evaluation_error": {"phase": "result_getter", "func": func_name, "type": type(exc).__name__, "message": str(exc)}}

        try:
            if expected_getter_cfg:
                expected_state = self._call_getter(expected_getter_cfg, env_proxy)
                reward = float(metric_fn(result_state, expected_state, **options))
            else:
                reward = float(metric_fn(result_state, **options))
        except Exception as exc:
            logger.exception("Unexpected error during metric/expected for %s", func_name)
            return 0.0, {**info, "evaluation_error": {"phase": "metric_or_expected", "func": func_name, "type": type(exc).__name__, "message": str(exc)}}
        return reward, info

    def _evaluate_multiple(self, evaluator_cfg: dict[str, Any], env_proxy: GuestAgentProxy) -> tuple[float, dict[str, Any]]:
        conj: str = evaluator_cfg.get("conj", "and")
        func_names: list[str] = evaluator_cfg["func"]
        info: dict[str, Any] = {"evaluator_func": func_names, "conj": conj}
        result_cfgs: list = evaluator_cfg.get("result") or [None] * len(func_names)
        expected_cfgs: list = evaluator_cfg.get("expected") or [None] * len(func_names)
        options_list: list[dict] = (
            [o or {} for o in evaluator_cfg["options"]]
            if isinstance(evaluator_cfg.get("options"), list)
            else [{}] * len(func_names)
        )

        results: list[float] = []
        sub_errors: list[dict[str, Any]] = []
        for func_name, result_cfg, expected_cfg, opts in zip(func_names, result_cfgs, expected_cfgs, options_list):
            try:
                metric_fn: Metric = getattr(metrics, func_name)
            except AttributeError as exc:
                sub_errors.append({"phase": "metric_lookup", "func": func_name, "type": "AttributeError", "message": str(exc)})
                if conj == "and":
                    return 0.0, {**info, "evaluation_error": sub_errors[-1], "sub_errors": sub_errors}
                results.append(0.0)
                continue

            try:
                result_state = self._call_getter(result_cfg, env_proxy) if result_cfg else None
            except FileNotFoundError as exc:
                sub_errors.append({"phase": "result_getter", "func": func_name, "type": "FileNotFoundError", "message": str(exc)})
                if conj == "and":
                    return 0.0, {**info, "file_not_found": True, "evaluation_error": sub_errors[-1], "sub_errors": sub_errors}
                results.append(0.0)
                continue
            except Exception as exc:
                logger.exception("Unexpected error during result getter for %s", func_name)
                sub_errors.append({"phase": "result_getter", "func": func_name, "type": type(exc).__name__, "message": str(exc)})
                if conj == "and":
                    return 0.0, {**info, "evaluation_error": sub_errors[-1], "sub_errors": sub_errors}
                results.append(0.0)
                continue

            try:
                if expected_cfg:
                    expected_state = self._call_getter(expected_cfg, env_proxy)
                    score = float(metric_fn(result_state, expected_state, **opts))
                else:
                    score = float(metric_fn(result_state, **opts))
            except Exception as exc:
                logger.exception("Unexpected error during metric/expected for %s", func_name)
                sub_errors.append({"phase": "metric_or_expected", "func": func_name, "type": type(exc).__name__, "message": str(exc)})
                if conj == "and":
                    return 0.0, {**info, "evaluation_error": sub_errors[-1], "sub_errors": sub_errors}
                results.append(0.0)
                continue

            if conj == "and" and score == 0.0:
                return sum(results + [0.0]) / (len(results) + 1) if results else 0.0, info
            if conj == "or" and score == 1.0:
                return 1.0, info
            results.append(score)

        if sub_errors:
            info["sub_errors"] = sub_errors
        if not results:
            return 0.0, info
        return (sum(results) / len(results) if conj == "and" else max(results)), info

    @staticmethod
    def _call_getter(cfg: dict[str, Any], env_proxy: GuestAgentProxy) -> Any:
        getter_fn: Getter = getattr(getters, "get_{:}".format(cfg["type"]))
        return getter_fn(env_proxy, cfg)
