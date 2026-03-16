.PHONY: help install update format lint test coverage hello debug xray

help:
	@echo "make install    - Install dependencies in editable mode"
	@echo "make update     - Update dependencies"
	@echo "make format     - Format code"
	@echo "make lint       - Lint and auto-fix"
	@echo "make test       - Run unit tests"
	@echo "make coverage   - Run tests with coverage report"
	@echo "make hello      - Run hello_miniwob recipe"
	@echo "make debug      - Run hello_miniwob recipe in debug mode"
	@echo "make xray       - Run AL2 XRay viewer in debug mode"

hello:
	@echo "🤖 Running hello_miniwob recipe"
	uv pip install -e "cubes/miniwob[debug]"
	uv run recipes/hello_miniwob.py

debug:
	@echo "🤖 Running hello_miniwob recipe in debug mode"
	uv pip install -e "cubes/miniwob[debug]"
	uv run recipes/hello_miniwob.py debug

xray:
	uv run ch-xray --debug

install:
	@echo "🚀 Installing dependencies"
	@echo "Install requires sudo permissions to install Playwright dependencies. You may be prompted for your password."
	uv sync --all-extras
	uv run playwright install chromium --with-deps

update:
	@echo "🔄 Updating dependencies"
	uv sync --all-extras --upgrade
	uv run playwright install chromium --with-deps

lint:
	@echo "🧹 Linting code"
	uv run ruff check --fix .
	uv run ruff format .

lint-check:
	@echo "🧹 Checking lint"
	uvx ruff check --diff .
	uvx ruff format --diff .

test: install
	@echo "🧪 Running unit tests"
	uv run pytest -n 10 tests/ -v

coverage:
	@echo "📊 Running tests with coverage report"
	uv run pytest tests/ --cov=cube_harness --cov-report=term-missing
