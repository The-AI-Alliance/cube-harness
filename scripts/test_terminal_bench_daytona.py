#!/usr/bin/env python
"""
Test Terminal-Bench task execution with Daytona.

This script:
1. Loads a task from the exported dataset
2. Extracts it to a temp directory
3. Creates a Daytona sandbox from its Dockerfile
4. Runs the oracle solution (solution.sh) to verify setup works
5. Runs the tests (run-tests.sh) to validate

Usage:
    uv run scripts/test_terminal_bench_daytona.py --task-id hello-world
"""

import argparse
import io
import tarfile
import tempfile
import time
from pathlib import Path

from datasets import load_from_disk
from daytona import CreateSandboxFromImageParams, Daytona, Image, Resources


def extract_task(task: dict, extract_path: Path) -> None:
    """Extract task archive to a directory."""
    extract_path.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(task["archive"]), mode="r:gz") as tar:
        tar.extractall(path=extract_path, filter="data")


def run_command(sandbox, command: str, timeout: int = 300) -> tuple[str, int]:
    """Run a command in the sandbox and return (output, exit_code)."""
    from uuid import uuid4

    from daytona import SessionExecuteRequest

    session_id = str(uuid4())

    try:
        sandbox.process.create_session(session_id)
        response = sandbox.process.execute_session_command(
            session_id,
            SessionExecuteRequest(command=f"bash -c {repr(command)}", run_async=True),
            timeout=timeout,
        )

        if response.cmd_id is None:
            return "[error] No command ID returned", -1

        # Poll for completion
        cmd = sandbox.process.get_session_command(session_id, response.cmd_id)
        start = time.time()
        while cmd.exit_code is None:
            if time.time() - start > timeout:
                return "[error] Timeout", -1
            time.sleep(1)
            cmd = sandbox.process.get_session_command(session_id, response.cmd_id)

        logs = sandbox.process.get_session_command_logs(session_id, response.cmd_id)
        stdout = logs.stdout.strip() if logs.stdout else ""
        stderr = logs.stderr.strip() if logs.stderr else ""

        output = stdout
        if stderr:
            output += f"\n[stderr]\n{stderr}"

        return output, cmd.exit_code

    except Exception as e:
        return f"[error] {e}", -1


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Terminal-Bench task with Daytona")
    parser.add_argument("--task-id", type=str, default="hello-world", help="Task ID to test")
    parser.add_argument("--dataset-path", type=str, default="./data/terminal_bench", help="Path to dataset")
    parser.add_argument("--run-solution", action="store_true", help="Run the oracle solution before tests")
    parser.add_argument("--timeout", type=int, default=300, help="Command timeout in seconds")
    args = parser.parse_args()

    # Load dataset
    print(f"[INFO] Loading dataset from {args.dataset_path}")
    ds = load_from_disk(args.dataset_path)

    # Find task
    task = None
    for t in ds:
        if t["task_id"] == args.task_id:
            task = t
            break

    if task is None:
        print(f"[ERROR] Task '{args.task_id}' not found")
        print(f"[INFO] Available tasks: {[t['task_id'] for t in ds][:10]}...")
        return

    print(f"[INFO] Found task: {task['task_id']}")
    print(f"  Category: {task['category']}")
    print(f"  Difficulty: {task['difficulty']}")
    print(f"  Timeout: {task['max_agent_timeout_sec']}s (agent), {task['max_test_timeout_sec']}s (test)")

    with tempfile.TemporaryDirectory() as tmpdir:
        task_path = Path(tmpdir) / task["task_id"]

        # Extract task
        print(f"\n[INFO] Extracting task to {task_path}")
        extract_task(task, task_path)

        # List extracted files
        print("[INFO] Extracted files:")
        for f in sorted(task_path.rglob("*")):
            if f.is_file():
                print(f"  {f.relative_to(task_path)}")

        # Create Daytona client
        print("\n[INFO] Creating Daytona sandbox...")
        daytona = Daytona()

        # Check for Dockerfile
        dockerfile_path = task_path / "Dockerfile"
        if dockerfile_path.exists():
            # Read Dockerfile to check if it uses Terminal-Bench base images
            dockerfile_content = dockerfile_path.read_text()
            uses_tbench_image = "ghcr.io/laude-institute" in dockerfile_content

            if uses_tbench_image:
                print("[INFO] Task uses Terminal-Bench base image, using python:3.13-slim instead")
                # Use standard Python image with necessary tools
                image = Image.base("python:3.13-slim").dockerfile_commands(
                    [
                        "RUN apt-get update && apt-get install -y --no-install-recommends curl git ca-certificates && rm -rf /var/lib/apt/lists/*",
                        "WORKDIR /app",
                    ]
                )
            else:
                print(f"[INFO] Building image from {dockerfile_path}")
                image = Image.from_dockerfile(str(dockerfile_path))
        else:
            print("[INFO] No Dockerfile found, using python:3.13-slim")
            image = Image.base("python:3.13-slim").dockerfile_commands(
                [
                    "RUN apt-get update && apt-get install -y --no-install-recommends curl git ca-certificates && rm -rf /var/lib/apt/lists/*",
                    "WORKDIR /app",
                ]
            )

        # Add the entire task directory to /app in the sandbox
        image = image.add_local_dir(str(task_path), "/app")

        # Create sandbox
        params = CreateSandboxFromImageParams(
            image=image,
            resources=Resources(cpu=2, memory=4, disk=10),
            auto_stop_interval=0,
            auto_delete_interval=0,
        )

        sandbox = None
        try:
            sandbox = daytona.create(params, timeout=300)
            print(f"[INFO] Sandbox created: {sandbox.id}")

            # Show working directory
            output, exit_code = run_command(sandbox, "pwd && ls -la", timeout=30)
            print(f"\n[INFO] Working directory contents:\n{output}")

            # Run solution if requested
            if args.run_solution:
                solution_path = task_path / "solution.sh"
                if solution_path.exists():
                    print("\n[INFO] Running oracle solution (solution.sh)...")
                    output, exit_code = run_command(sandbox, "cd /app && bash solution.sh", timeout=args.timeout)
                    print(f"[OUTPUT]\n{output[:2000]}")
                    print(f"[EXIT CODE] {exit_code}")
                else:
                    print("[WARN] No solution.sh found")

            # Run tests
            run_tests_path = task_path / "run-tests.sh"
            if run_tests_path.exists():
                print("\n[INFO] Running tests (run-tests.sh)...")

                # Copy tests directory to container at TEST_DIR location
                output, exit_code = run_command(
                    sandbox,
                    "export TEST_DIR=/tests && mkdir -p $TEST_DIR && cp -r /app/tests/* $TEST_DIR/ && cd /app && bash run-tests.sh",
                    timeout=args.timeout,
                )
                print(f"[OUTPUT]\n{output[:3000]}")
                print(f"\n[EXIT CODE] {exit_code}")

                if exit_code == 0:
                    print("\n[SUCCESS] Tests passed!")
                else:
                    print("\n[FAILURE] Tests failed!")
            else:
                print("[WARN] No run-tests.sh found")

        finally:
            if sandbox:
                print("\n[INFO] Cleaning up sandbox...")
                daytona.delete(sandbox)
                print("[INFO] Done")


if __name__ == "__main__":
    main()
