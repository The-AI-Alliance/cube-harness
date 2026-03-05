.PHONY: test install

test:
	uv run python -m workarena_cube.debug

install:
	uv sync --all-extras
	uv pip install -e .
