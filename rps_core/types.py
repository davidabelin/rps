from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Literal


class Action(IntEnum):
    ROCK = 0
    PAPER = 1
    SCISSORS = 2


ActionLike = int | str | Action


def normalize_action(value: ActionLike) -> Action:
    if isinstance(value, Action):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"rock", "r"}:
            return Action.ROCK
        if lowered in {"paper", "p"}:
            return Action.PAPER
        if lowered in {"scissors", "scissor", "s"}:
            return Action.SCISSORS
        if lowered.isdigit():
            value = int(lowered)
        else:
            raise ValueError(f"Unknown action string: {value}")
    if int(value) not in (0, 1, 2):
        raise ValueError(f"Action must be 0, 1, or 2, got: {value}")
    return Action(int(value))


Outcome = Literal["player", "ai", "tie"]


@dataclass(slots=True)
class RoundObservation:
    step: int
    last_opponent_action: int | None
    cumulative_reward: int


@dataclass(slots=True)
class RoundTransition:
    observation: RoundObservation
    action: int
    opponent_action: int
    outcome: Outcome
    reward_delta: int
    round_index: int


@dataclass(slots=True)
class RoundResult:
    player_action: int
    opponent_action: int
    outcome: Outcome
    reward_delta: int
    round_index: int


@dataclass(slots=True)
class GameState:
    game_id: int
    rounds_played: int
    score_player: int
    score_ai: int
    score_ties: int
    session_index: int
