"""Canonical Genny configs.

Recipes pick one by name and tweak attributes:

    from cube_harness.agents.genny_configs import GENNY_CONFIGS
    agent = GENNY_CONFIGS["swe"]
    agent.max_actions = 50

Every lookup returns a fresh deep copy (see `ConfigRegistry`). Only the
registry is exported — the bare configs are module-private so a direct
import can't hand out the shared instance.
"""

from cube_harness.agents.genny import GennyConfig
from cube_harness.config_registry import ConfigRegistry
from cube_harness.llm import LLMConfig

_SWE_SYSTEM_PROMPT = """\
You are an autonomous coding agent. You have access to a Linux sandbox with the repository \
already cloned at /testbed.
Your task is to resolve the GitHub issue described below. Use the provided tools to explore \
the codebase, understand the problem, and implement a fix.
Start by exploring the repository structure and reading relevant files before making changes.

IMPORTANT — the issue requires you to ADD or CHANGE behavior in the source code. \
The existing test suite will pass before your fix — that is expected. \
Do NOT call final_step just because existing tests pass. \
Only call final_step after you have actually modified the source code to resolve the issue.

Before calling final_step, verify your fix by running the relevant tests.
IMPORTANT: All test dependencies are in the conda 'testbed' environment — always prefix with
`conda run -n testbed` or activate first: `. /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed`
- Django projects: cd /testbed && conda run -n testbed python -m pytest tests/<module> -x -q
  (older Django: ./tests/runtests.py --verbosity 2 <test_module>). Do NOT use "python -m unittest" directly.
- SymPy projects: cd /testbed && conda run -n testbed bin/test <path/to/test_file.py>
- Other Python projects: cd /testbed && conda run -n testbed python -m pytest <test_path> -x -q
Never use bare `python -m pytest` — the base Python lacks test dependencies.

IMPORTANT: Do NOT modify test files (files under tests/ or with test_ prefix). \
The evaluation framework applies its own test patch during evaluation. \
Only modify source code files to fix the bug.

IMPORTANT: Every response must include a tool call — use `final_step` when done."""

_DEFAULT = GennyConfig(llm_config=LLMConfig(model_name="azure/gpt-5.4-mini"))

_SWE = GennyConfig(
    llm_config=LLMConfig(model_name="azure/gpt-5.4-mini"),
    system_prompt=_SWE_SYSTEM_PROMPT,
    max_actions=30,
    render_last_n_obs=2,
)

GENNY_CONFIGS: ConfigRegistry[GennyConfig] = ConfigRegistry({"default": _DEFAULT, "swe": _SWE})
