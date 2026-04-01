.PHONY: test install

test:
	uv run cube test terminalbench-cube

install:
	uv sync --all-extras
	uv pip install -e .
