from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from random import Random

import numpy as np

from rps_agents import build_heuristic_agent
from rps_core.scoring import score_round
from rps_core.types import RoundObservation, RoundTransition
from rps_storage.object_store import write_bytes


def _state_index(last_opponent_action: int | None) -> int:
    if last_opponent_action is None:
        return 3
    return int(last_opponent_action)


@dataclass(slots=True)
class RLTrainConfig:
    episodes: int = 300
    steps_per_episode: int = 300
    alpha: float = 0.15
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    seed: int = 7
    opponents: tuple[str, ...] = (
        "rock",
        "paper",
        "scissors",
        "copy_opponent",
        "reactionary",
        "counter_reactionary",
        "statistical",
        "markov",
    )


def train_q_policy(config: RLTrainConfig, artifact_path: str) -> dict:
    rng = Random(config.seed)
    np_rng = np.random.default_rng(config.seed)
    q_table = np.zeros((4, 3), dtype=float)
    epsilon = float(config.epsilon_start)
    episode_rewards: list[float] = []
    episode_win_rates: list[float] = []

    for episode in range(config.episodes):
        opponent_name = config.opponents[episode % len(config.opponents)]
        opponent = build_heuristic_agent(opponent_name)
        opponent.reset(seed=rng.randint(0, 2**31 - 1))
        opponent_obs = RoundObservation(step=0, last_opponent_action=None, cumulative_reward=0)

        total_reward = 0.0
        wins = 0
        losses = 0
        ties = 0
        last_opp_action: int | None = None

        for step in range(config.steps_per_episode):
            state = _state_index(last_opp_action)
            if np_rng.random() < epsilon:
                action = int(np_rng.integers(0, 3))
            else:
                action = int(np.argmax(q_table[state]))

            opponent_action = int(opponent.select_action(opponent_obs))
            reward = score_round(action, opponent_action)
            next_state = _state_index(opponent_action)
            td_target = reward + config.gamma * float(np.max(q_table[next_state]))
            q_table[state, action] += config.alpha * (td_target - q_table[state, action])

            if reward > 0:
                wins += 1
            elif reward < 0:
                losses += 1
            else:
                ties += 1
            total_reward += reward

            opponent.observe(
                RoundTransition(
                    observation=opponent_obs,
                    action=opponent_action,
                    opponent_action=action,
                    outcome="player" if reward < 0 else "ai" if reward > 0 else "tie",
                    reward_delta=-reward,
                    round_index=step,
                )
            )
            opponent_obs = RoundObservation(
                step=step + 1,
                last_opponent_action=action,
                cumulative_reward=opponent_obs.cumulative_reward - reward,
            )
            last_opp_action = opponent_action

        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)
        episode_rewards.append(total_reward / max(1, config.steps_per_episode))
        episode_win_rates.append(wins / max(1, wins + losses))

    policy = np.argmax(q_table, axis=1).astype(int)
    metrics = {
        "episodes": config.episodes,
        "steps_per_episode": config.steps_per_episode,
        "mean_episode_reward": float(np.mean(episode_rewards)),
        "final_50_episode_reward": float(np.mean(episode_rewards[-50:] if len(episode_rewards) >= 50 else episode_rewards)),
        "mean_non_tie_win_rate": float(np.mean(episode_win_rates)),
        "final_50_non_tie_win_rate": float(
            np.mean(episode_win_rates[-50:] if len(episode_win_rates) >= 50 else episode_win_rates)
        ),
        "q_table": q_table.tolist(),
        "policy": policy.tolist(),
    }
    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "model_type": "rl_qtable",
        "config": asdict(config),
        "q_table": q_table.tolist(),
        "policy": policy.tolist(),
    }
    write_bytes(artifact_path, pickle.dumps(artifact), content_type="application/octet-stream")
    metrics["artifact_path"] = artifact_path
    return metrics
