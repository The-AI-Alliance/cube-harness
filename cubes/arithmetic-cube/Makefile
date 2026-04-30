.PHONY: test install

test:
	uv run cube test arithmetic-cube

install:
	uv sync --all-extras
	pip install -e .