from __future__ import annotations

from rps_core.types import Action, Outcome, normalize_action

ACTION_NAMES = {
    int(Action.ROCK): "rock",
    int(Action.PAPER): "paper",
    int(Action.SCISSORS): "scissors",
}


def score_round(left: int, right: int) -> int:
    left_action = int(normalize_action(left))
    right_action = int(normalize_action(right))
    if left_action == right_action:
        return 0
    return 1 if (left_action - right_action) % 3 == 1 else -1


def counter_action(action: int) -> int:
    return (int(normalize_action(action)) + 1) % 3


def reward_to_outcome(player_reward_delta: int) -> Outcome:
    if player_reward_delta > 0:
        return "player"
    if player_reward_delta < 0:
        return "ai"
    return "tie"
