"""Opponent transition-matrix heuristic."""

from __future__ import annotations

import numpy as np

from rps_agents.heuristic.common import RNGMixin
from rps_core.types import RoundObservation, RoundTransition


class OpponentTransitionMatrixAgent(RNGMixin):
    """Estimate opponent move transitions and counter sampled prediction."""

    name = "opponent_transition_matrix"

    def __init__(self) -> None:
        """Initialize transition matrix and previous-opponent state."""

        super().__init__()
        self.transition_counts = np.zeros((3, 3), dtype=float)
        self.prev_opp: int | None = None

    def reset(self, seed: int | None) -> None:
        """Reset RNG and clear transition statistics."""

        super().reset(seed)
        self.transition_counts = np.zeros((3, 3), dtype=float)
        self.prev_opp = None

    def select_action(self, obs: RoundObservation) -> int:
        """Sample opponent next move from transition row and counter it."""

        if obs.last_opponent_action is None or self.prev_opp is None:
            return self._rand_action()
        row = self.transition_counts[int(obs.last_opponent_action)]
        if row.sum() <= 0:
            return self._rand_action()
        probs = row / row.sum()
        pred = int(np.random.choice([0, 1, 2], p=probs))
        return int((pred + 1) % 3)

    def observe(self, transition: RoundTransition) -> None:
        """Update transition counts from observed opponent sequence."""

        opponent = int(transition.opponent_action)
        if self.prev_opp is not None:
            self.transition_counts[self.prev_opp, opponent] += 1.0
        self.prev_opp = opponent
