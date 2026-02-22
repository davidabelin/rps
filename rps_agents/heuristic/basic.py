from __future__ import annotations

from collections import Counter

from rps_agents.heuristic.common import RNGMixin
from rps_core.scoring import counter_action, score_round
from rps_core.types import RoundObservation, RoundTransition


class CopyOpponentAgent(RNGMixin):
    name = "copy_opponent"

    def __init__(self) -> None:
        super().__init__()

    def select_action(self, obs: RoundObservation) -> int:
        if obs.last_opponent_action is None:
            return self._rand_action()
        return int(obs.last_opponent_action)

    def observe(self, transition: RoundTransition) -> None:
        return None


class ReactionaryAgent(RNGMixin):
    name = "reactionary"

    def __init__(self) -> None:
        super().__init__()
        self._last_action: int | None = None

    def reset(self, seed: int | None) -> None:
        super().reset(seed)
        self._last_action = None

    def select_action(self, obs: RoundObservation) -> int:
        if self._last_action is None:
            self._last_action = self._rand_action()
            return self._last_action
        if obs.last_opponent_action is not None and score_round(self._last_action, obs.last_opponent_action) <= 0:
            self._last_action = counter_action(obs.last_opponent_action)
        return self._last_action

    def observe(self, transition: RoundTransition) -> None:
        self._last_action = transition.action


class CounterReactionaryAgent(RNGMixin):
    name = "counter_reactionary"

    def __init__(self) -> None:
        super().__init__()
        self._last_action: int | None = None

    def reset(self, seed: int | None) -> None:
        super().reset(seed)
        self._last_action = None

    def select_action(self, obs: RoundObservation) -> int:
        if self._last_action is None:
            self._last_action = self._rand_action()
            return self._last_action
        if obs.last_opponent_action is not None and score_round(self._last_action, obs.last_opponent_action) == 1:
            self._last_action = (self._last_action + 2) % 3
        elif obs.last_opponent_action is not None:
            self._last_action = counter_action(obs.last_opponent_action)
        return self._last_action

    def observe(self, transition: RoundTransition) -> None:
        self._last_action = transition.action


class StatisticalAgent(RNGMixin):
    name = "statistical"

    def __init__(self) -> None:
        super().__init__()
        self._counts: Counter[int] = Counter()

    def reset(self, seed: int | None) -> None:
        super().reset(seed)
        self._counts.clear()

    def select_action(self, obs: RoundObservation) -> int:
        if not self._counts:
            return self._rand_action()
        mode_action = self._counts.most_common(1)[0][0]
        return counter_action(mode_action)

    def observe(self, transition: RoundTransition) -> None:
        self._counts[int(transition.opponent_action)] += 1
