.PHONY: help install update format lint test coverage hello debug viewer xray

help:
	@echo "make install    - Install dependencies in editable mode"
	@echo "make update     - Update dependencies"
	@echo "make format     - Format code"
	@echo "make lint       - Lint and auto-fix"
	@echo "make test       - Run unit tests"
	@echo "make coverage   - Run tests with coverage report"
	@echo "make hello      - Run hello_miniwob recipe"
	@echo "make debug      - Run hello_miniwob recipe in debug mode"
	@echo "make viewer     - Run AL2 experiment viewer in debug mode"
	@echo "make xray       - Run AL2 XRay viewer in debug mode"

hello:
	uv run recipes/hello_miniwob.py

debug:
	uv run recipes/hello_miniwob.py debug

viewer:
	uv run ch-viewer --debug

xray:
	uv run ch-xray --debug

install:
	uv sync --all-extras
	uv run playwright install chromium --with-deps

update:
	uv sync --all-extras --upgrade
	uv run playwright install chromium --with-deps

lint:
	uv run ruff check --fix .
	uv run ruff format .

lint-check:
	uvx ruff check --diff .
	uvx ruff format --diff .

test: install
	uv run pytest -n 10 tests/ -v

coverage:
	uv run pytest tests/ --cov=cube_harness --cov-report=term-missing
