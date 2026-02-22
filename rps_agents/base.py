from __future__ import annotations

from typing import Protocol

from rps_core.types import RoundObservation, RoundTransition


class AgentProtocol(Protocol):
    name: str

    def reset(self, seed: int | None) -> None:
        ...

    def select_action(self, obs: RoundObservation) -> int:
        ...

    def observe(self, transition: RoundTransition) -> None:
        ...
