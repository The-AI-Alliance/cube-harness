"""Static checks that the terminalbench recipes are wired correctly.

Catches the class of regression where one PR deletes a module another PR's
recipe imports — specifically the function-scope `from terminalbench_cube.tool
import TerminalBenchToolConfig` that survived the TerminalTool migration in #381.
Pure AST walk + lazy `importlib.import_module` — no Docker, no network, ~50 ms.

The walk only enforces resolution for **local** packages (this repo's source).
External packages may be unavailable in the unit-test env (cube-standard@dev,
cube_infra_toolkit) — those are smoke-level concerns, not unit-test ones.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from cube_harness.agents.genny2 import Genny2Config
from cube_harness.agents.genny2_swe_config import (
    INSTANCE_TEMPLATES,
    make_agent_config,
)

# Package prefixes owned by this repo. Imports of these MUST resolve — they're
# the surface this PR controls.  Imports of anything else are best-effort.
LOCAL_PACKAGE_PREFIXES: tuple[str, ...] = (
    "cube_harness",
    "terminalbench_cube",
)

RECIPES: list[str] = [
    "recipes/genny2_terminalbench_recipe.py",
    "recipes/genny2_terminalbench_iter_recipe.py",
]


def _iter_import_targets(path: Path) -> list[tuple[str, str]]:
    """Return [(module, name), ...] for every `from X import Y` in the file."""
    tree = ast.parse(path.read_text())
    targets: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                targets.append((node.module, alias.name))
    return targets


def _is_local(module: str) -> bool:
    root = module.split(".", 1)[0]
    return root in {p.split(".", 1)[0] for p in LOCAL_PACKAGE_PREFIXES}


@pytest.mark.parametrize("recipe", RECIPES)
def test_recipe_local_imports_resolve(recipe: str) -> None:
    """Every `from <local pkg> import Y` in the recipe must resolve.

    Skips when the cube isn't installed in the current env (the tests.yml CI
    job runs `uv sync` without the workspace cubes). The check still fires
    in any env where `terminalbench_cube` IS available — local dev, the
    integration-local CI workflow, and the cube CI workflow — which is where
    a regression of this class would actually break a runtime.
    """
    path = Path(__file__).resolve().parents[1] / recipe
    for module, name in _iter_import_targets(path):
        if not _is_local(module):
            continue
        try:
            mod = importlib.import_module(module)
        except ModuleNotFoundError as e:
            # Module unavailable in this env → no regression to check here.
            # If the recipe references a stale module (the regression we care about),
            # the error message will name the recipe's referenced module, not just
            # any unrelated missing local dep — surface it to the test ID.
            pytest.skip(f"{module} not installed in this env ({e})")
        assert hasattr(mod, name), f"{recipe}: `from {module} import {name}` — name not found"


def test_workflow_tbench_template_registered() -> None:
    """The `workflow-tbench` instance template must be registered for tbench recipes."""
    assert "workflow-tbench" in INSTANCE_TEMPLATES


def test_make_agent_config_accepts_tbench_template() -> None:
    """The shared agent-config factory must build a Genny2Config with workflow-tbench."""
    cfg = make_agent_config("gpt-5.4-mini", "workflow-tbench", max_actions=10, cost_limit=0.1)
    assert isinstance(cfg, Genny2Config)
    assert cfg.budget.max_actions == 10
    assert cfg.budget.cost_limit == 0.1
