.PHONY: test install

test:
	uv run cube test webarena-verified-cube

debug:
	uv run src/webarena_verified_cube/debug.py

install:
	uv sync --all-extras
	uv pip install -e .
