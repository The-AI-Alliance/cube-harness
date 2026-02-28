"""Agent implementations for AgentLab2."""

from agentlab2.agents.genny import Genny, GennyConfig
from agentlab2.agents.legacy_generic_agent import (
    GenericAgent,
    GenericAgentConfig,
    GenericPromptFlags,
)
from agentlab2.agents.react import ReactAgent, ReactAgentConfig

__all__ = [
    "Genny",
    "GennyConfig",
    "GenericAgent",
    "GenericAgentConfig",
    "GenericPromptFlags",
    "ReactAgent",
    "ReactAgentConfig",
]
