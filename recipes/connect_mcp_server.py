"""Demo: Connect an MCP server and use it as a tool with a ReAct agent.

This recipe shows how to:
1. Configure an MCP server (the official filesystem server)
2. Create an MCPTool that connects to it and discovers available tools
3. Run a simple agent loop where the ReAct agent uses MCP tools

Prerequisites:
    npm (or npx) must be available on PATH.
    Set your LLM API key in .env (e.g. OPENAI_API_KEY or ANTHROPIC_API_KEY).

Usage:
    uv run recipes/hello_mcp.py
"""

import logging
import tempfile
from pathlib import Path

from cube.core import Observation
from cube.task import STOP_ACTION
from dotenv import load_dotenv
from termcolor import colored

from cube_harness.agents.react import ReactAgentConfig
from cube_harness.llm import LLMConfig
from cube_harness.tools.mcp import MCPServerConfig, MCPToolConfig

load_dotenv()

LOG_FORMAT = "[%(levelname)s] %(asctime)s - %(name)s:%(lineno)d - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

MAX_STEPS = 10


def setup_demo_directory(demo_dir: Path) -> None:
    """Seed the demo directory with a few files for the agent to discover."""
    demo_dir.mkdir(parents=True, exist_ok=True)

    (demo_dir / "readme.txt").write_text(
        "Welcome to cube-harness!\nThis is a demo directory managed by the MCP filesystem server.\n"
    )
    (demo_dir / "shopping_list.txt").write_text("eggs\nbread\nmilk\napples\n")
    notes = demo_dir / "notes"
    notes.mkdir(exist_ok=True)
    (notes / "todo.txt").write_text("1. Try MCP tools\n2. Build an agent\n")


def main() -> None:
    # -- 1. Prepare a temp directory with demo files --
    demo_dir = Path(tempfile.mkdtemp(prefix="cube_harness_mcp_"))
    setup_demo_directory(demo_dir)
    logger.info(f"Demo directory: {demo_dir}")

    # -- 2. Configure the MCP filesystem server --
    # Uses the official @modelcontextprotocol/server-filesystem package.
    # The server is scoped to `demo_dir` so the agent can only access files there.
    mcp_config = MCPToolConfig(
        servers={
            "filesystem": MCPServerConfig(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", str(demo_dir)],
            ),
        },
    )

    # -- 3. Create the tool and connect --
    tool = mcp_config.make()
    tool.reset()  # connects to the MCP server and discovers tools

    discovered = tool.action_set
    logger.info(f"Discovered {len(discovered)} MCP tools:")
    for schema in discovered:
        logger.info(f"  - {schema.name}: {schema.description}")

    # -- 4. Set up the ReAct agent with MCP tools --
    llm_config = LLMConfig(model_name="azure/gpt-5-mini", temperature=1.0)
    agent_config = ReactAgentConfig(llm_config=llm_config, max_actions=MAX_STEPS)
    agent = agent_config.make(action_set=discovered)

    # -- 5. Run a simple agent loop --
    goal = (
        "You have access to a filesystem at protected folder. "
        "Please do the following:\n"
        "1. List the contents of the root directory.\n"
        "2. Read the shopping_list.txt file.\n"
        "3. Add 'butter' to the shopping list.\n"
        "4. Create a new file called 'summary.txt' with a one-line summary of what you found.\n"
        "When done, call final_step to finish."
    )
    obs = Observation.from_text(goal)

    logger.info(f"Goal: {goal}")
    for step in range(MAX_STEPS):
        agent_output = agent.step(obs)
        logger.info(f"Step {step + 1}: Agent produced {len(agent_output.actions)} action(s)")

        # Execute each action through the MCP tool
        done = False
        for action in agent_output.actions:
            if action.name == STOP_ACTION.name:
                logger.info("Agent called final_step — stopping.")
                done = True
                break
            logger.info(colored(f"Action {action})", "magenta"))
            obs = tool.execute_action(action)
            text = str(obs)[:500]
            logger.info(colored(f"Obs {text}", "blue"))

        if done:
            break

    # -- 6. Show what changed --
    logger.info("--- Final directory contents ---")
    for path in sorted(demo_dir.rglob("*")):
        if path.is_file():
            logger.info(f"  {path.relative_to(demo_dir)}: {path.read_text().strip()[:120]}")

    # -- 7. Clean up --
    tool.close()
    logger.info("Done!")


if __name__ == "__main__":
    main()
