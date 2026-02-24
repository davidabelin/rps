"""Core typed structures shared across simulation, API, and training code.

Terminology note
----------------
This project uses dual terminology depending on context:

- Gameplay/UI context: ``game`` + ``round``
- RL/training context: ``episode`` + ``step``

In code, ``RoundObservation.step`` can be interpreted as:

- the round index within a game session, or
- the step index within an episode.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Literal


class Action(IntEnum):
    """Canonical integer encoding for Rock-Paper-Scissors actions."""

    ROCK = 0
    PAPER = 1
    SCISSORS = 2


ActionLike = int | str | Action


def normalize_action(value: ActionLike) -> Action:
    """Normalize user/model action input into the canonical ``Action`` enum.

    Parameters
    ----------
    value : ActionLike
        Input action value in one of these forms:

        - integer ``0``, ``1``, ``2``
        - action enum ``Action.ROCK`` / ``PAPER`` / ``SCISSORS``
        - action string (``"rock"``, ``"paper"``, ``"scissors"``), including
          short aliases (``"r"``, ``"p"``, ``"s"``)

    Returns
    -------
    Action
        Canonical enum value.

    Raises
    ------
    ValueError
        If the input cannot be interpreted as a valid action.
    """

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
    """Observation presented to an agent before action selection.

    Parameters
    ----------
    step : int
        Zero-based position in the current session (round index / episode step).
    last_opponent_action : int | None
        Opponent's previous action, or ``None`` at the first step.
    cumulative_reward : int
        Reward accumulated so far from this agent's perspective.
    """

    step: int
    last_opponent_action: int | None
    cumulative_reward: int


@dataclass(slots=True)
class RoundTransition:
    """Transition recorded after both players have acted for one step.

    Parameters
    ----------
    observation : RoundObservation
        Observation used for action selection at this step.
    action : int
        Action taken by the current agent.
    opponent_action : int
        Action taken by the opponent.
    outcome : Outcome
        Discrete outcome label (``"player"``, ``"ai"``, ``"tie"``).
    reward_delta : int
        Immediate reward from this agent's perspective (``-1``/``0``/``1``).
    round_index : int
        Step index where this transition occurred.
    """

    observation: RoundObservation
    action: int
    opponent_action: int
    outcome: Outcome
    reward_delta: int
    round_index: int


@dataclass(slots=True)
class RoundResult:
    """Round-level result returned to API/UI callers.

    Parameters
    ----------
    player_action : int
        Action selected by the human/external caller.
    opponent_action : int
        Action selected by the AI/opponent.
    outcome : Outcome
        Outcome from the player's perspective.
    reward_delta : int
        Immediate player reward (``-1``, ``0``, ``1``).
    round_index : int
        Round/step index in the current session.
    """

    player_action: int
    opponent_action: int
    outcome: Outcome
    reward_delta: int
    round_index: int


@dataclass(slots=True)
class GameState:
    """Aggregate score/state for a human-vs-AI game session."""

    game_id: int
    rounds_played: int
    score_player: int
    score_ai: int
    score_ties: int
    session_index: int
