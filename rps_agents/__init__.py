"""Agent registry and factory functions."""

from rps_agents.base import AgentProtocol
from rps_agents.heuristic import AGENT_SPECS, AgentSpec, build_heuristic_agent, list_agent_specs
from rps_agents.model_agent import ModelBackedAgent

__all__ = [
    "AgentProtocol",
    "AgentSpec",
    "AGENT_SPECS",
    "build_heuristic_agent",
    "list_agent_specs",
    "ModelBackedAgent",
]
