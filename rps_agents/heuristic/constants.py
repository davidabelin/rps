from __future__ import annotations

from dataclasses import dataclass

from rps_agents.heuristic.common import RNGMixin
from rps_core.types import RoundObservation, RoundTransition


@dataclass
class ConstantAgent(RNGMixin):
    name: str
    action: int

    def __post_init__(self) -> None:
        RNGMixin.__init__(self)

    def select_action(self, obs: RoundObservation) -> int:
        return int(self.action)

    def observe(self, transition: RoundTransition) -> None:
        return None


class NashEquilibriumAgent(RNGMixin):
    name = "nash_equilibrium"

    def __init__(self) -> None:
        super().__init__()

    def select_action(self, obs: RoundObservation) -> int:
        return self._rand_action()

    def observe(self, transition: RoundTransition) -> None:
        return None


class HitLastOwnActionAgent(RNGMixin):
    name = "hit_last_own_action"

    def __init__(self) -> None:
        super().__init__()
        self._last_action = 0

    def reset(self, seed: int | None) -> None:
        super().reset(seed)
        self._last_action = 0

    def select_action(self, obs: RoundObservation) -> int:
        self._last_action = (self._last_action + 1) % 3
        return self._last_action

    def observe(self, transition: RoundTransition) -> None:
        return None
