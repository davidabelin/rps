from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from random import Random
from typing import Callable

from rps_agents.base import AgentProtocol
from rps_core.scoring import score_round
from rps_core.types import RoundObservation, RoundTransition

AgentFactory = Callable[[], AgentProtocol]


@dataclass(slots=True)
class EpisodeResult:
    score_a: int
    score_b: int
    ties: int


def play_episode(agent_a: AgentProtocol, agent_b: AgentProtocol, steps: int, seed: int | None = None) -> EpisodeResult:
    rng = Random(seed)
    agent_a.reset(seed=rng.randint(0, 2**31 - 1))
    agent_b.reset(seed=rng.randint(0, 2**31 - 1))
    score_a = 0
    score_b = 0
    ties = 0
    obs_a = RoundObservation(step=0, last_opponent_action=None, cumulative_reward=0)
    obs_b = RoundObservation(step=0, last_opponent_action=None, cumulative_reward=0)
    for step in range(steps):
        action_a = int(agent_a.select_action(obs_a))
        action_b = int(agent_b.select_action(obs_b))
        reward_a = score_round(action_a, action_b)
        reward_b = -reward_a
        if reward_a > 0:
            score_a += 1
        elif reward_a < 0:
            score_b += 1
        else:
            ties += 1
        agent_a.observe(
            RoundTransition(
                observation=obs_a,
                action=action_a,
                opponent_action=action_b,
                outcome="player" if reward_a > 0 else "ai" if reward_a < 0 else "tie",
                reward_delta=reward_a,
                round_index=step,
            )
        )
        agent_b.observe(
            RoundTransition(
                observation=obs_b,
                action=action_b,
                opponent_action=action_a,
                outcome="player" if reward_b > 0 else "ai" if reward_b < 0 else "tie",
                reward_delta=reward_b,
                round_index=step,
            )
        )
        obs_a = RoundObservation(step=step + 1, last_opponent_action=action_b, cumulative_reward=obs_a.cumulative_reward + reward_a)
        obs_b = RoundObservation(step=step + 1, last_opponent_action=action_a, cumulative_reward=obs_b.cumulative_reward + reward_b)
    return EpisodeResult(score_a=score_a, score_b=score_b, ties=ties)


def run_round_robin(agent_factories: dict[str, AgentFactory], episodes: int, steps: int, seed: int) -> list[dict]:
    rng = Random(seed)
    agent_names = sorted(agent_factories.keys())
    rows: list[dict] = []
    for left_index, left_name in enumerate(agent_names):
        for right_name in agent_names[left_index + 1 :]:
            total_a = 0
            total_b = 0
            total_ties = 0
            for _ in range(episodes):
                result = play_episode(
                    agent_factories[left_name](),
                    agent_factories[right_name](),
                    steps=steps,
                    seed=rng.randint(0, 2**31 - 1),
                )
                total_a += result.score_a
                total_b += result.score_b
                total_ties += result.ties
            rounds_played = episodes * steps
            rows.append(
                {
                    "agent_a": left_name,
                    "agent_b": right_name,
                    "agent_a_win_rate": total_a / rounds_played,
                    "agent_b_win_rate": total_b / rounds_played,
                    "draw_rate": total_ties / rounds_played,
                }
            )
    return rows


def leaderboard(rows: list[dict]) -> list[dict]:
    aggregate = defaultdict(lambda: {"wins": 0.0, "losses": 0.0, "draws": 0.0, "matches": 0})
    for row in rows:
        a = row["agent_a"]
        b = row["agent_b"]
        aggregate[a]["wins"] += row["agent_a_win_rate"]
        aggregate[a]["losses"] += row["agent_b_win_rate"]
        aggregate[a]["draws"] += row["draw_rate"]
        aggregate[a]["matches"] += 1
        aggregate[b]["wins"] += row["agent_b_win_rate"]
        aggregate[b]["losses"] += row["agent_a_win_rate"]
        aggregate[b]["draws"] += row["draw_rate"]
        aggregate[b]["matches"] += 1
    table = []
    for name, stats in aggregate.items():
        matches = max(1, stats["matches"])
        table.append(
            {
                "agent": name,
                "avg_win_rate": stats["wins"] / matches,
                "avg_loss_rate": stats["losses"] / matches,
                "avg_draw_rate": stats["draws"] / matches,
            }
        )
    table.sort(key=lambda row: row["avg_win_rate"], reverse=True)
    return table
