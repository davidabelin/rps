from __future__ import annotations

"""Agent protocol definitions shared by heuristic and model-backed agents."""

from typing import Protocol

from rps_core.types import RoundObservation, RoundTransition


class AgentProtocol(Protocol):
    """Minimal interface required by the RPS engine.

    Implementations are expected to maintain internal state across calls for a
    single game session.
    """

    name: str

    def reset(self, seed: int | None) -> None:
        """Reset internal state before starting a new game/episode."""

        ...

    def select_action(self, obs: RoundObservation) -> int:
        """Select an action for the current observation.

        Parameters
        ----------
        obs : RoundObservation
            Current observation at this step.

        Returns
        -------
        int
            Action encoded as ``0`` (rock), ``1`` (paper), ``2`` (scissors).
        """

        ...

    def observe(self, transition: RoundTransition) -> None:
        """Update internal state from the resulting transition."""

        ...
