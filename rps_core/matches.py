"""Agent-vs-agent match helpers for replayable RPS sessions.

Role
----
Provide the neutral replay generator used by arena jobs, non-persisted match
API endpoints, and any later tournament-style evaluation work.
"""

from __future__ import annotations

from random import Random
from typing import Callable

from rps_agents.base import AgentProtocol
from rps_core.scoring import ACTION_NAMES, score_round
from rps_core.types import RoundObservation, RoundTransition, normalize_action


def _winner_from_reward(reward_delta: int) -> str:
    """Map agent-A reward sign to the arena winner label."""

    if reward_delta > 0:
        return "agent_a"
    if reward_delta < 0:
        return "agent_b"
    return "tie"


def _transition_outcome(reward_delta: int) -> str:
    """Map reward sign into the player/ai/tie labels expected by agents."""

    if reward_delta > 0:
        return "player"
    if reward_delta < 0:
        return "ai"
    return "tie"


def play_agent_match(
    *,
    agent_a: AgentProtocol,
    agent_b: AgentProtocol,
    agent_a_name: str,
    agent_b_name: str,
    rounds: int,
    seed: int | None = None,
    on_round: Callable[[dict], None] | None = None,
) -> dict:
    """Run one replayable RPS match between two agents.

    Role
    ----
    Execute the canonical agent-vs-agent match loop while emitting a trace that
    can be persisted, streamed, or replayed by the web arena.

    Parameters
    ----------
    on_round : callable | None
        Optional callback invoked after each resolved round with a JSON-ready
        frame payload.

    Returns
    -------
    dict
        Match summary plus full replay trace.

    Used By
    -------
    ``rps_web.match_jobs.MatchJobManager`` and ``rps_web.blueprints.game``.
    """

    if int(rounds) <= 0:
        raise ValueError("rounds must be a positive integer.")

    seed_rng = Random(seed)
    agent_a.reset(seed_rng.randrange(0, 2**31) if seed is not None else None)
    agent_b.reset(seed_rng.randrange(0, 2**31) if seed is not None else None)

    obs_a = RoundObservation(step=0, last_opponent_action=None, cumulative_reward=0)
    obs_b = RoundObservation(step=0, last_opponent_action=None, cumulative_reward=0)

    score_agent_a = 0
    score_agent_b = 0
    score_ties = 0
    trace: list[dict] = []

    for round_index in range(int(rounds)):
        action_a = int(normalize_action(agent_a.select_action(obs_a)))
        action_b = int(normalize_action(agent_b.select_action(obs_b)))
        reward_a = int(score_round(action_a, action_b))
        reward_b = -reward_a
        winner = _winner_from_reward(reward_a)

        if reward_a > 0:
            score_agent_a += 1
        elif reward_a < 0:
            score_agent_b += 1
        else:
            score_ties += 1

        agent_a.observe(
            RoundTransition(
                observation=obs_a,
                action=action_a,
                opponent_action=action_b,
                outcome=_transition_outcome(reward_a),
                reward_delta=reward_a,
                round_index=round_index,
            )
        )
        agent_b.observe(
            RoundTransition(
                observation=obs_b,
                action=action_b,
                opponent_action=action_a,
                outcome=_transition_outcome(reward_b),
                reward_delta=reward_b,
                round_index=round_index,
            )
        )

        obs_a = RoundObservation(
            step=round_index + 1,
            last_opponent_action=action_b,
            cumulative_reward=obs_a.cumulative_reward + reward_a,
        )
        obs_b = RoundObservation(
            step=round_index + 1,
            last_opponent_action=action_a,
            cumulative_reward=obs_b.cumulative_reward + reward_b,
        )

        frame = {
            "round_index": round_index,
            "agent_a_action": action_a,
            "agent_b_action": action_b,
            "agent_a_action_name": ACTION_NAMES[action_a],
            "agent_b_action_name": ACTION_NAMES[action_b],
            "winner": winner,
            "reward_agent_a": reward_a,
            "reward_agent_b": reward_b,
            "score_agent_a": score_agent_a,
            "score_agent_b": score_agent_b,
            "score_ties": score_ties,
        }
        trace.append(frame)
        if on_round is not None:
            on_round(dict(frame))

    match_winner = "tie"
    if score_agent_a > score_agent_b:
        match_winner = "agent_a"
    elif score_agent_b > score_agent_a:
        match_winner = "agent_b"

    return {
        "mode": "agent_vs_agent",
        "agent_a": agent_a_name,
        "agent_b": agent_b_name,
        "rounds": int(rounds),
        "seed": seed,
        "winner": match_winner,
        "score_agent_a": score_agent_a,
        "score_agent_b": score_agent_b,
        "score_ties": score_ties,
        "trace": trace,
    }
