"""Constant/simple mixed-strategy baseline agents."""

from __future__ import annotations

from dataclasses import dataclass

from rps_agents.heuristic.common import RNGMixin
from rps_core.types import RoundObservation, RoundTransition


@dataclass
class ConstantAgent(RNGMixin):
    """Always return a fixed action id."""

    name: str
    action: int

    def __post_init__(self) -> None:
        """Initialize mixin state in dataclass construction flow."""

        RNGMixin.__init__(self)

    def select_action(self, obs: RoundObservation) -> int:
        """Return the configured constant action."""

        return int(self.action)

    def observe(self, transition: RoundTransition) -> None:
        """No-op: policy has no adaptive state."""

        return None


class NashEquilibriumAgent(RNGMixin):
    """Uniform random policy (mixed-strategy equilibrium for RPS)."""

    name = "nash_equilibrium"

    def __init__(self) -> None:
        """Initialize RNG state."""

        super().__init__()

    def select_action(self, obs: RoundObservation) -> int:
        """Sample uniformly random action."""

        return self._rand_action()

    def observe(self, transition: RoundTransition) -> None:
        """No-op: policy remains stationary."""

        return None


class HitLastOwnActionAgent(RNGMixin):
    """Cycle through own actions to attack copy-like opponents."""

    name = "hit_last_own_action"

    def __init__(self) -> None:
        """Initialize cycle state."""

        super().__init__()
        self._last_action = 0

    def reset(self, seed: int | None) -> None:
        """Reset RNG and cycle pointer."""

        super().reset(seed)
        self._last_action = 0

    def select_action(self, obs: RoundObservation) -> int:
        """Advance action cycle and return current action."""

        self._last_action = (self._last_action + 1) % 3
        return self._last_action

    def observe(self, transition: RoundTransition) -> None:
        """No-op: next action is determined only by cycle state."""

        return None
