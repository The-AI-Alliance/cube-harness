.PHONY: test install

test:
	uv run python -m miniwob_cube.debug

install:
	uv sync --all-extras
	pip install -e .
