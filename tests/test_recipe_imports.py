"""Static-analysis guard: every `from <local pkg> import Y` in any recipe resolves.

Catches the class of bug that broke #381's consolidated branch — one PR deleted
`terminalbench2_cube.tool`, another's recipe still imported it from inside a
function. The regression only surfaced at runtime on a non-debug code path.

Pure AST walk + lazy `importlib.import_module` — no Docker, no network, no LLM
calls. The walk skips recipes/cube-specific paths whose top-level package isn't
installed in this env (tests.yml CI runs `uv sync` without the workspace cubes);
the cube CI workflow and any local dev env that has the cube installed picks up
the assertion.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

# Package prefixes owned by this repo. Imports of these MUST resolve.  Imports
# from external packages (cube_infra_*, anthropic, etc.) are best-effort and
# may live behind optional dependency groups — skip when unavailable.
LOCAL_PACKAGE_PREFIXES: tuple[str, ...] = (
    "cube_harness",
    # Cubes under cubes/* (workspace packages installed by `make install`).
    "arithmetic_cube",
    "browsercomp",
    "miniwob",
    "osworld_cube",
    "swebench_live_cube",
    "swebench_verified_cube",
    "terminalbench2_cube",
    "webarena_verified_cube",
    "waa_cube",
)

REPO_ROOT: Path = Path(__file__).resolve().parents[1]


def _recipes() -> list[Path]:
    """Return all recipe .py files under recipes/, excluding venvs and dunders."""
    return [
        p
        for p in REPO_ROOT.glob("recipes/**/*.py")
        if "venv" not in p.parts and not p.name.startswith("_") and p.name != "__init__.py"
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
    return root in LOCAL_PACKAGE_PREFIXES


@pytest.mark.parametrize("recipe", _recipes(), ids=lambda p: p.relative_to(REPO_ROOT).as_posix())
def test_recipe_local_imports_resolve(recipe: Path) -> None:
    """Every `from <local pkg> import Y` in the recipe must resolve."""
    for module, name in _iter_import_targets(recipe):
        if not _is_local(module):
            continue
        try:
            mod = importlib.import_module(module)
        except ModuleNotFoundError as e:
            # Local package unavailable in this env (e.g. tests.yml CI doesn't
            # install workspace cubes) — no regression to assert here.
            pytest.skip(f"{module} not installed in this env ({e})")
        assert hasattr(mod, name), f"{recipe.relative_to(REPO_ROOT)}: `from {module} import {name}` — name not found"
