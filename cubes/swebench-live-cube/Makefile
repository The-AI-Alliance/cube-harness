.PHONY: test install

test:
	uv run cube test swebench-live-cube

install:
	uv sync --all-extras
	uv pip install -e .
