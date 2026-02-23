from __future__ import annotations

"""Baseline reactive heuristic agents for gameplay and benchmarking."""

from collections import Counter

from rps_agents.heuristic.common import RNGMixin
from rps_core.scoring import counter_action, score_round
from rps_core.types import RoundObservation, RoundTransition


class CopyOpponentAgent(RNGMixin):
    """Play the opponent's previous action.

    Notes
    -----
    Uses random action on the first round when no previous opponent action is
    available.
    """

    name = "copy_opponent"

    def __init__(self) -> None:
        """Initialize agent RNG state."""

        super().__init__()

    def select_action(self, obs: RoundObservation) -> int:
        """Select mirrored action from observation history."""

        if obs.last_opponent_action is None:
            return self._rand_action()
        return int(obs.last_opponent_action)

    def observe(self, transition: RoundTransition) -> None:
        """No internal update required for this policy."""

        return None


class ReactionaryAgent(RNGMixin):
    """Repeat wins; counter opponent after tie/loss.

    The policy keeps its previous action when it beats the last opponent move.
    Otherwise it switches to the counter of the opponent's last action.
    """

    name = "reactionary"

    def __init__(self) -> None:
        """Initialize internal previous-action state."""

        super().__init__()
        self._last_action: int | None = None

    def reset(self, seed: int | None) -> None:
        """Reset RNG and clear remembered action."""

        super().reset(seed)
        self._last_action = None

    def select_action(self, obs: RoundObservation) -> int:
        """Select next action under reactionary logic."""

        if self._last_action is None:
            self._last_action = self._rand_action()
            return self._last_action
        if obs.last_opponent_action is not None and score_round(self._last_action, obs.last_opponent_action) <= 0:
            self._last_action = counter_action(obs.last_opponent_action)
        return self._last_action

    def observe(self, transition: RoundTransition) -> None:
        """Record the action used for next-step reaction."""

        self._last_action = transition.action


class CounterReactionaryAgent(RNGMixin):
    """Meta-policy designed to exploit reactionary-style opponents."""

    name = "counter_reactionary"

    def __init__(self) -> None:
        """Initialize internal previous-action state."""

        super().__init__()
        self._last_action: int | None = None

    def reset(self, seed: int | None) -> None:
        """Reset RNG and clear remembered action."""

        super().reset(seed)
        self._last_action = None

    def select_action(self, obs: RoundObservation) -> int:
        """Select next action under counter-reactionary logic."""

        if self._last_action is None:
            self._last_action = self._rand_action()
            return self._last_action
        if obs.last_opponent_action is not None and score_round(self._last_action, obs.last_opponent_action) == 1:
            self._last_action = (self._last_action + 2) % 3
        elif obs.last_opponent_action is not None:
            self._last_action = counter_action(obs.last_opponent_action)
        return self._last_action

    def observe(self, transition: RoundTransition) -> None:
        """Record the latest own action."""

        self._last_action = transition.action


class StatisticalAgent(RNGMixin):
    """Counter the opponent's empirical mode action."""

    name = "statistical"

    def __init__(self) -> None:
        """Initialize action-count statistics."""

        super().__init__()
        self._counts: Counter[int] = Counter()

    def reset(self, seed: int | None) -> None:
        """Reset RNG and clear action-frequency table."""

        super().reset(seed)
        self._counts.clear()

    def select_action(self, obs: RoundObservation) -> int:
        """Choose the counter of the most frequent observed opponent move."""

        if not self._counts:
            return self._rand_action()
        mode_action = self._counts.most_common(1)[0][0]
        return counter_action(mode_action)

    def observe(self, transition: RoundTransition) -> None:
        """Update opponent action frequencies."""

        self._counts[int(transition.opponent_action)] += 1
