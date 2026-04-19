.PHONY: debug install test debug debug-docker

install:
	uv sync --all-extras
	uv pip install -e .
	uv run python scripts/create_task_metadata.py

test:
	uv run pytest tests/

debug:
	uv run cube test osworld-cube

debug-docker:
	uv run cube test osworld_cube.debug_docker
