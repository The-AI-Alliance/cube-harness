"""The one line every recipe ends with.

A recipe is a declarative config file — it builds one or more `Experiment`
objects and hands them to `run`:

    from cube_harness.recipe import run

    exp = Experiment(name="...", agent_config=..., benchmark_config=...)

    if __name__ == "__main__":
        run(exp)                 # or run(exp_small, exp_large)

`run` ships a fixed, generic CLI — identical for every recipe, not
extensible per-recipe (clone the file for anything structural):

    python recipes/foo.py                       # Ray, full benchmark
    python recipes/foo.py --limit 3             # in-process, first 3 tasks
    python recipes/foo.py --ray 32              # override Ray worker count
    python recipes/foo.py --set agent_config.max_actions=200 --set name=tweaked
"""

import json
from typing import Annotated

import typer

from cube_harness.exp_runner import run_sequentially, run_with_ray
from cube_harness.experiment import Experiment


def _apply_override(exp: Experiment, dotted: str) -> None:
    """Apply one ``a.b.c=value`` override. Value is JSON-parsed when possible
    (so ``200`` → int), then assigned — `ValidatedConfig` validates the type."""
    path, _, raw = dotted.partition("=")
    if not _:
        raise typer.BadParameter(f"--set expects key=value, got {dotted!r}")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    *parents, leaf = path.split(".")
    target = exp
    for part in parents:
        target = getattr(target, part)
    setattr(target, leaf, value)


def run(*exps: Experiment) -> None:
    """Run one or more experiments through the generic recipe CLI."""
    if not exps:
        raise ValueError("run() needs at least one Experiment")

    def _main(
        limit: Annotated[int | None, typer.Option(help="Run first N tasks in-process (no Ray) — debug.")] = None,
        ray: Annotated[int, typer.Option(help="Ray worker count (ignored with --limit).")] = 8,
        set_: Annotated[
            list[str] | None, typer.Option("--set", help="Override: dotted.path=value (repeatable).")
        ] = None,
    ) -> None:
        for exp in exps:
            for override in set_ or []:
                _apply_override(exp, override)
            if limit is not None:
                run_sequentially(exp, debug_limit=limit)
            else:
                run_with_ray(exp, n_cpus=ray)

    typer.run(_main)
