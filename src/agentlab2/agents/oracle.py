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
    """Oracle agent that runs solution.sh to solve Terminal-Bench tasks.

    This agent is used for validation - it should achieve 100% accuracy
    since it runs the reference solution.
    """

    def __init__(self, config: OracleAgentConfig, action_set: list[ActionSchema]) -> None:
        self.config = config
        self.action_set = action_set
        self._step_count = 0
        self._solution_run = False

    def step(self, obs: Observation) -> AgentOutput:
        """Run solution.sh on first step, then call final_step."""
        self._step_count += 1

        if not self._solution_run:
            # First step: run the solution
            self._solution_run = True
            logger.info("Oracle agent running solution.sh")
            return AgentOutput(
                actions=[
                    Action(
                        name="bash",
                        arguments={"command": "cd /app && bash solution.sh", "timeout": 300},
                    )
                ]
            )
        else:
            # Second step: signal completion
            logger.info("Oracle agent calling final_step")
            return AgentOutput(actions=[Action(name="final_step", arguments={})])
