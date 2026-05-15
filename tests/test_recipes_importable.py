"""Canonical recipes must stay importable and define a valid Experiment.

This guards against config-schema drift: when a config field changes, a stale
recipe fails here instead of at someone's next run. Recipes are PEP-723
scripts whose cube/tool deps are not in the base test env, so a recipe is
skipped (not failed) when only its cube is missing — a broken `cube_harness`
import still fails, which is the signal we want.
"""

import importlib.util
from pathlib import Path

import pytest

from cube_harness.experiment import Experiment

RECIPES = sorted((Path(__file__).parent.parent / "recipes").glob("*.py"))
_CUBE_DEPS = {"miniwob_cube", "webarena_verified_cube", "swebench_verified_cube", "cube_browser_tool"}


@pytest.mark.parametrize("recipe", RECIPES, ids=lambda p: p.name)
def test_recipe_imports_and_defines_experiment(recipe: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cube_harness.experiment.make_experiment_output_dir", lambda *a, **k: tmp_path)
    spec = importlib.util.spec_from_file_location(recipe.stem, recipe)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as e:
        if (e.name or "").split(".")[0] in _CUBE_DEPS:
            pytest.skip(f"recipe cube dependency not installed: {e.name}")
        raise
    exps: list[Experiment] = []
    for v in vars(module).values():
        if isinstance(v, Experiment):
            exps.append(v)
        elif isinstance(v, dict):
            exps.extend(x for x in v.values() if isinstance(x, Experiment))
    assert exps, f"{recipe.name} defines no Experiment (a var or a dict[str, Experiment])"
