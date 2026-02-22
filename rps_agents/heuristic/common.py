from __future__ import annotations

from random import Random


class RNGMixin:
    def __init__(self) -> None:
        self._rng = Random()

    def reset(self, seed: int | None) -> None:
        self._rng.seed(seed)

    def _rand_action(self) -> int:
        return self._rng.randrange(0, 3)
