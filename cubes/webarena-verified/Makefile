.PHONY: test install

test:
	uv run cube test webarena-verified-cube

install:
	uv sync --all-extras
	uv pip install -e .
