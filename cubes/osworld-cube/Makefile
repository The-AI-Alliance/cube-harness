.PHONY: debug install test debug debug-docker

install:
	uv sync --all-extras
	pip install -e .

test:
	uv run pytest tests/

debug:
	uv run cube test osworld_cube

debug-docker:
	uv run cube test osworld_cube.debug_docker
