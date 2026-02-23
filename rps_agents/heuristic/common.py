from __future__ import annotations

"""Common mixins/utilities shared by heuristic agents."""

from random import Random


class RNGMixin:
    """Provide deterministic RNG behavior for heuristic agents.

    Notes
    -----
    Agents inheriting this mixin should call ``reset(seed)`` at episode/game
    boundaries so evaluation remains reproducible.
    """

    def __init__(self) -> None:
        """Initialize private random number generator."""

        self._rng = Random()

    def reset(self, seed: int | None) -> None:
        """Reset RNG state from optional seed."""

        self._rng.seed(seed)

    def _rand_action(self) -> int:
        """Sample a random action id in ``{0, 1, 2}``."""

        return self._rng.randrange(0, 3)
