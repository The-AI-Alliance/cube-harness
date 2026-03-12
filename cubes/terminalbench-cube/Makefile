.PHONY: test install

test:
	uv run python -m terminalbench_cube.debug

install:
	uv sync --all-extras
	uv pip install -e .
