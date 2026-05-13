#!/usr/bin/env python3
"""Smoke test: terminalbench-cube debug suite on real Docker.

Exercises plumbing the unit tests can't reach — Task.reset() (container health
check, archive extract, /app→/tmp/app rewrite), Task.evaluate() (test upload,
uv pre-install), Task.close() — by running both oracle debug tasks through
LocalInfraConfig + Docker. Equivalent to `cube test terminalbench-cube` but
wrapped in the cube-harness smoke contract.

Auto-skips if the Docker daemon is unreachable.

Final line follows the cube-harness smoke contract:
    SMOKE OK: terminalbench_debug_suite    (exit 0)
    SMOKE FAIL: terminalbench_debug_suite  (exit 1)
    SMOKE SKIP: terminalbench_debug_suite  (exit 2)

Usage:
    uv run scripts/smoke/terminalbench_debug_suite.py
"""

from __future__ import annotations

import logging
import shutil
import socket
from typing import Annotated

import typer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s %(message)s")

NAME = "terminalbench_debug_suite"


def _docker_available() -> bool:
    """True iff `docker` is on PATH and the daemon answers."""
    if shutil.which("docker") is None:
        return False
    try:
        # Daemon liveness — `docker info` requires the socket to be reachable.
        import subprocess

        proc = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, OSError, socket.error):
        return False


def main(
    keep_containers: Annotated[bool, typer.Option(help="Keep containers after the run (for debugging)")] = False,
) -> None:
    """Run the terminalbench-cube debug suite and report smoke OK/FAIL/SKIP."""
    if not _docker_available():
        typer.echo(f"SMOKE SKIP: {NAME} — docker daemon unreachable")
        raise typer.Exit(2)

    try:
        import terminalbench_cube.debug as _tbench_debug
        from cube.infra_local import LocalInfraConfig
        from cube.testing import run_debug_suite
    except ImportError as e:
        typer.echo(f"SMOKE SKIP: {NAME} — import error ({e})")
        raise typer.Exit(2) from e

    results = run_debug_suite(
        "terminalbench-cube",
        _tbench_debug,
        workers=1,
        infra=LocalInfraConfig(),
    )

    failed = [r for r in results if r["error"] or not r["done"] or r["reward"] < 1.0]
    typer.echo(f"\n{len(results) - len(failed)}/{len(results)} debug tasks reached reward=1.0")
    for r in results:
        marker = "PASS" if r in [x for x in results if x not in failed] else "FAIL"
        typer.echo(f"  {marker}  {r.get('task_id', '<unknown>'):24s} reward={r.get('reward')}")

    if failed:
        typer.echo(f"SMOKE FAIL: {NAME}")
        raise typer.Exit(1)
    typer.echo(f"SMOKE OK: {NAME}")
    if keep_containers:
        typer.echo("(containers left running — clean up manually with `docker ps` / `docker rm`)")
    raise typer.Exit(0)


if __name__ == "__main__":
    typer.run(main)
