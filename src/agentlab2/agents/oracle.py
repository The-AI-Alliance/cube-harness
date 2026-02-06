"""Oracle agent that runs the reference solution.

Used for validating benchmark setup - should achieve 100% accuracy.
"""

import logging

from agentlab2.agent import Agent, AgentConfig
from agentlab2.core import Action, ActionSchema, AgentOutput, Observation

logger = logging.getLogger(__name__)


class OracleAgentConfig(AgentConfig):
    """Config for Oracle agent."""

    def make(self, action_set: list[ActionSchema]) -> "OracleAgent":
        return OracleAgent(config=self, action_set=action_set)


class OracleAgent(Agent):
    """Oracle agent that runs the reference solution for Terminal-Bench tasks.

    This agent is used for validation - it should achieve 100% accuracy
    since it runs the reference solution.

    Supports both TB1 (solution.sh) and TB2 (solution/solve.sh) formats.
    """

    def __init__(self, config: OracleAgentConfig, action_set: list[ActionSchema]) -> None:
        self.config = config
        self.action_set = action_set
        self._step_count = 0
        self._solution_run = False

    def step(self, obs: Observation) -> AgentOutput:
        """Run solution on first step, then call final_step."""
        self._step_count += 1

        if not self._solution_run:
            # First step: run the solution
            # TB2 uploads solution to /solution, TB1 has solution.sh in /app
            self._solution_run = True
            logger.info("Oracle agent running solution")
            return AgentOutput(
                actions=[
                    Action(
                        name="bash",
                        arguments={
                            "command": "cd /app && if [ -f /solution/solve.sh ]; then bash /solution/solve.sh; elif [ -f solution.sh ]; then bash solution.sh; else echo 'No solution found'; fi",
                            "timeout": 600,
                        },
                    )
                ]
            )
        else:
            # Second step: signal completion
            logger.info("Oracle agent calling final_step")
            return AgentOutput(actions=[Action(name="final_step", arguments={})])
