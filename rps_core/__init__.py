"""Core Rock-Paper-Scissors game primitives."""

from .scoring import ACTION_NAMES, counter_action, score_round
from .types import Action, GameState, RoundObservation, RoundResult, RoundTransition, normalize_action

__all__ = [
    "Action",
    "GameState",
    "RoundObservation",
    "RoundResult",
    "RoundTransition",
    "normalize_action",
    "ACTION_NAMES",
    "counter_action",
    "score_round",
]
