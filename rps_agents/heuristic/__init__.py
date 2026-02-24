"""Registry and factory functions for heuristic RPS agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from rps_agents.base import AgentProtocol
from rps_agents.heuristic.basic import CopyOpponentAgent, CounterReactionaryAgent, ReactionaryAgent, StatisticalAgent
from rps_agents.heuristic.constants import ConstantAgent, HitLastOwnActionAgent, NashEquilibriumAgent
from rps_agents.heuristic.decision_tree import DecisionTreeHeuristicAgent
from rps_agents.heuristic.ensemble import PollingAgent, RotatingEnsembleAgent
from rps_agents.heuristic.markov import MarkovAgent
from rps_agents.heuristic.memory_patterns import MemoryPatternAgent
from rps_agents.heuristic.multi_armed_bandit import MultiArmedBanditAgent
from rps_agents.heuristic.opponent_transition import OpponentTransitionMatrixAgent


@dataclass(frozen=True)
class AgentSpec:
    """Metadata descriptor for one registered heuristic agent."""

    name: str
    description: str
    factory: Callable[[], AgentProtocol]


def _build_specs() -> dict[str, AgentSpec]:
    """Build static heuristic-agent registry."""

    return {
        "rock": AgentSpec("rock", "Always play rock.", lambda: ConstantAgent("rock", 0)),
        "paper": AgentSpec("paper", "Always play paper.", lambda: ConstantAgent("paper", 1)),
        "scissors": AgentSpec("scissors", "Always play scissors.", lambda: ConstantAgent("scissors", 2)),
        "copy_opponent": AgentSpec("copy_opponent", "Mirror the opponent's last move.", CopyOpponentAgent),
        "reactionary": AgentSpec("reactionary", "Counter the opponent after weak outcomes.", ReactionaryAgent),
        "counter_reactionary": AgentSpec(
            "counter_reactionary", "Counter strategy tuned against reactionary opponents.", CounterReactionaryAgent
        ),
        "statistical": AgentSpec("statistical", "Counter the opponent's most frequent move.", StatisticalAgent),
        "nash_equilibrium": AgentSpec("nash_equilibrium", "Uniform random mixed strategy.", NashEquilibriumAgent),
        "hit_last_own_action": AgentSpec(
            "hit_last_own_action", "Cycle own action to attack predicted copy-cat play.", HitLastOwnActionAgent
        ),
        "markov": AgentSpec("markov", "Markov chain predictor over action sequence windows.", MarkovAgent),
        "memory_patterns": AgentSpec("memory_patterns", "Pattern memory over recent mixed action history.", MemoryPatternAgent),
        "opponent_transition_matrix": AgentSpec(
            "opponent_transition_matrix",
            "Transition matrix over opponent moves.",
            OpponentTransitionMatrixAgent,
        ),
        "decision_tree": AgentSpec(
            "decision_tree",
            "Online decision tree heuristic over local/global rollouts.",
            DecisionTreeHeuristicAgent,
        ),
        "multi_armed_bandit": AgentSpec(
            "multi_armed_bandit",
            "Thompson-sampling ensemble over multiple predictors.",
            MultiArmedBanditAgent,
        ),
        "polling_agent": AgentSpec("polling_agent", "Weighted voting ensemble over multiple agents.", PollingAgent),
        "rotating_ensemble": AgentSpec(
            "rotating_ensemble",
            "Randomly rotates among voter agents on prime intervals.",
            RotatingEnsembleAgent,
        ),
    }


AGENT_SPECS = _build_specs()


def list_agent_specs() -> list[AgentSpec]:
    """Return all registered heuristic specs sorted by name."""

    return [AGENT_SPECS[name] for name in sorted(AGENT_SPECS.keys())]


def build_heuristic_agent(name: str) -> AgentProtocol:
    """Instantiate one heuristic agent by registry name."""

    if name not in AGENT_SPECS:
        raise KeyError(f"Unknown heuristic agent: {name}")
    return AGENT_SPECS[name].factory()
