"""Source-path resolution and experiment config loading."""

from __future__ import annotations

import importlib.util
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cube_harness.experiment import Experiment

logger = logging.getLogger(__name__)


def _resolve_module_root(dotted_name: str) -> Path | None:
    """Map a `_type` value (e.g. 'swebench_verified_cube.benchmark.X') to a directory."""
    top = dotted_name.split(".")[0]
    try:
        spec = importlib.util.find_spec(top)
    except (ImportError, ValueError):
        return None
    if spec is None or spec.origin is None:
        return None
    origin = Path(spec.origin)
    return origin.parent if origin.name == "__init__.py" else origin.parent


@dataclass
class _ExperimentView:
    """A view of `experiment_config.json` that prefers the typed `Experiment` object
    but falls back to a raw dict when `_type` references can't be imported (e.g.
    the experiment was run with an agent class that no longer exists locally)."""

    experiment: Experiment | None
    raw: dict[str, Any]

    @property
    def agent_dotted(self) -> str:
        if self.experiment is not None:
            t = type(self.experiment.agent_config)
            return f"{t.__module__}.{t.__name__}"
        return self.raw.get("agent_config", {}).get("_type", "unknown")

    @property
    def benchmark_dotted(self) -> str:
        if self.experiment is not None:
            t = type(self.experiment.benchmark_config)
            return f"{t.__module__}.{t.__name__}"
        return self.raw.get("benchmark_config", {}).get("_type", "unknown")

    @property
    def infra_dotted(self) -> str | None:
        if self.experiment is not None and self.experiment.infra is not None:
            t = type(self.experiment.infra)
            return f"{t.__module__}.{t.__name__}"
        return self.raw.get("infra", {}).get("_type") if isinstance(self.raw.get("infra"), dict) else None


def _load_experiment_view(path: Path) -> _ExperimentView:
    """Load experiment_config.json. Try typed Experiment first; fall back to dict.

    Typed load can fail in many ways: a referenced class was renamed
    (`ImportError`), removed (`AttributeError`, when `_type` is `__main__.X` from
    an ad-hoc script), or changed shape (`ValidationError`). Any failure falls
    back to the dict view, which is always good enough for the judge — it only
    needs the `_type` strings and a few well-known fields.
    """
    if not path.exists():
        return _ExperimentView(experiment=None, raw={})
    raw = json.loads(path.read_text())
    try:
        return _ExperimentView(experiment=Experiment.load_config(str(path)), raw=raw)
    except Exception as e:
        logger.info(
            "experiment_config.json could not be loaded as Experiment (%s: %s) — "
            "falling back to dict view. Source paths will still be resolved from `_type` strings.",
            type(e).__name__,
            e,
        )
        return _ExperimentView(experiment=None, raw=raw)


def collect_source_paths(view: _ExperimentView) -> dict[str, Path]:
    """Resolve the on-disk source paths the judge should be able to grep.

    Returns a name→path map (only paths that exist). Includes the cube package
    referenced by the benchmark, the agent package, and the cube-standard /
    cube-harness installs.
    """
    paths: dict[str, Path] = {}

    def _add(name: str, dotted: str | None) -> None:
        if not dotted:
            return
        root = _resolve_module_root(dotted)
        if root is not None and root.exists():
            paths[name] = root

    _add("cube_package", view.benchmark_dotted)
    _add("agent_package", view.agent_dotted)
    _add("infra_package", view.infra_dotted)
    _add("cube_harness", "cube_harness.eval_log")
    _add("cube_standard", "cube.core")
    return paths
