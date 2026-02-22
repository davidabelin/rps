from __future__ import annotations

from random import Random

from rps_agents.heuristic.basic import CopyOpponentAgent, CounterReactionaryAgent, ReactionaryAgent, StatisticalAgent
from rps_agents.heuristic.decision_tree import DecisionTreeHeuristicAgent
from rps_agents.heuristic.markov import MarkovAgent
from rps_agents.heuristic.opponent_transition import OpponentTransitionMatrixAgent
from rps_core.types import RoundObservation, RoundTransition


class RotatingEnsembleAgent:
    name = "rotating_ensemble"

    def __init__(self) -> None:
        self._rng = Random()
        self._primes = [7, 11, 13, 17, 19, 23, 29, 31]
        self._voters = [
            CopyOpponentAgent(),
            ReactionaryAgent(),
            CounterReactionaryAgent(),
            StatisticalAgent(),
            OpponentTransitionMatrixAgent(),
            MarkovAgent(),
            DecisionTreeHeuristicAgent(),
        ]
        self._period = self._primes[0]
        self._current_index = 0

    def reset(self, seed: int | None) -> None:
        self._rng.seed(seed)
        self._period = self._rng.choice(self._primes)
        self._current_index = self._rng.randrange(0, len(self._voters))
        for voter in self._voters:
            voter.reset(seed)

    def select_action(self, obs: RoundObservation) -> int:
        if obs.step < 3:
            return 0
        if obs.step % self._period == 0:
            self._period = self._rng.choice(self._primes)
            self._current_index = self._rng.randrange(0, len(self._voters))
        action = int(self._voters[self._current_index].select_action(obs))
        return action if action in (0, 1, 2) else self._rng.randrange(0, 3)

    def observe(self, transition: RoundTransition) -> None:
        for voter in self._voters:
            voter.observe(transition)


class PollingAgent:
    name = "polling_agent"

    def __init__(self) -> None:
        self._voters = [
            CopyOpponentAgent(),
            ReactionaryAgent(),
            CounterReactionaryAgent(),
            StatisticalAgent(),
            OpponentTransitionMatrixAgent(),
            MarkovAgent(),
            DecisionTreeHeuristicAgent(),
        ]

    def reset(self, seed: int | None) -> None:
        for voter in self._voters:
            voter.reset(seed)

    def select_action(self, obs: RoundObservation) -> int:
        if obs.step < 3:
            return 0
        votes = [int(agent.select_action(obs)) for agent in self._voters]
        vote_counts = [votes.count(0), votes.count(1), votes.count(2)]
        decision_tree_vote = votes[-1] if votes else 0
        vote_counts[decision_tree_vote] += 5
        return int(max(range(3), key=lambda idx: vote_counts[idx]))

    def observe(self, transition: RoundTransition) -> None:
        for voter in self._voters:
            voter.observe(transition)
