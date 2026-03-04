.PHONY: test install

test:
	uv run python -m arithmetic_cube.debug

install:
	uv sync --all-extras
	uv pip install -e .