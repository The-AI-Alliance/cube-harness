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
    DEFAULT_TBENCH_TEMPLATE,
    make_tbench_agent_config,
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
    """Every `from <local pkg> import Y` in the recipe must resolve."""
    path = Path(__file__).resolve().parents[1] / recipe
    for module, name in _iter_import_targets(path):
        if not _is_local(module):
            continue
        mod = importlib.import_module(module)
        assert hasattr(mod, name), f"{recipe}: `from {module} import {name}` — name not found"


def test_make_tbench_agent_config_constructs() -> None:
    """The tbench agent-config factory must produce a valid Genny2Config."""
    cfg = make_tbench_agent_config("gpt-5.4-mini", DEFAULT_TBENCH_TEMPLATE, max_actions=10, cost_limit=0.1)
    assert isinstance(cfg, Genny2Config)
    assert cfg.budget.max_actions == 10
    assert cfg.budget.cost_limit == 0.1
