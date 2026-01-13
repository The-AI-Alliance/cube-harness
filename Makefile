.PHONY: help install format lint test coverage hello debug

help:
	@echo "make install    - Install dependencies in editable mode"
	@echo "make format     - Format code"
	@echo "make lint       - Lint and auto-fix"
	@echo "make test       - Run unit tests"
	@echo "make coverage   - Run tests with coverage report"
	@echo "make hello      - Run hello_miniwob recipe"
	@echo "make debug      - Run hello_miniwob recipe in debug mode"

hello:
	uv run recipes/hello_miniwob.py

debug:
	uv run recipes/hello_miniwob.py debug

install:
	uv sync --all-extras
	uv pip install -e .
	uv run playwright install chromium --with-deps

format:
	uv run ruff format .

lint:
	uv run ruff check --fix .

test: install
	uv run pytest -n 10 tests/ -v

coverage:
	uv run pytest tests/ --cov=agentlab2 --cov-report=term-missing
