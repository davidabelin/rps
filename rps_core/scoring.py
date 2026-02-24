"""Scoring helpers for classic 3-sign Rock-Paper-Scissors."""

from __future__ import annotations

from rps_core.types import Action, Outcome, normalize_action

ACTION_NAMES = {
    int(Action.ROCK): "rock",
    int(Action.PAPER): "paper",
    int(Action.SCISSORS): "scissors",
}


def score_round(left: int, right: int) -> int:
    """Score one RPS action pair.

    Parameters
    ----------
    left : int
        Action from the left/player-A side.
    right : int
        Action from the right/player-B side.

    Returns
    -------
    int
        Reward from ``left`` perspective:

        - ``1`` if ``left`` wins
        - ``0`` for tie
        - ``-1`` if ``left`` loses
    """

    left_action = int(normalize_action(left))
    right_action = int(normalize_action(right))
    if left_action == right_action:
        return 0
    return 1 if (left_action - right_action) % 3 == 1 else -1


def counter_action(action: int) -> int:
    """Return the action that beats the provided action."""

    return (int(normalize_action(action)) + 1) % 3


def reward_to_outcome(player_reward_delta: int) -> Outcome:
    """Convert numeric player reward into a symbolic outcome label."""

    if player_reward_delta > 0:
        return "player"
    if player_reward_delta < 0:
        return "ai"
    return "tie"
