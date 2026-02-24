"""CLI entry point for tabular RL policy training."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rps_rl.trainer import RLTrainConfig, train_q_policy
from rps_storage import RPSRepository
from rps_storage.object_store import is_gcs_uri, join_storage_path


def main() -> int:
    """Run RL training CLI and register resulting model artifact."""

    parser = argparse.ArgumentParser(description="Train a tabular RL policy (Q-learning) for RPS.")
    parser.add_argument("--db-path", type=str, default="data/rps.db")
    parser.add_argument("--models-dir", type=str, default="data/models")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--steps-per-episode", type=int, default=300)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--opponents",
        type=str,
        default="rock,paper,scissors,copy_opponent,reactionary,counter_reactionary,statistical,markov",
    )
    args = parser.parse_args()

    repository = RPSRepository(args.db_path)
    repository.init_schema()
    opponents = tuple(item.strip() for item in args.opponents.split(",") if item.strip())
    config = RLTrainConfig(
        episodes=args.episodes,
        steps_per_episode=args.steps_per_episode,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
        opponents=opponents,
    )

    models_dir = str(args.models_dir)
    if not is_gcs_uri(models_dir):
        Path(models_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    artifact_path = join_storage_path(models_dir, f"rl_qtable_cli_{timestamp}.pkl")
    metrics = train_q_policy(config=config, artifact_path=artifact_path)
    model_row = repository.create_model(
        name=f"rl-qtable-cli-{timestamp}",
        model_type="rl_qtable",
        artifact_path=artifact_path,
        lookback=1,
        metrics=metrics,
    )
    print("RL training completed")
    print(f"Model ID: {model_row['id']}")
    print(f"Artifact: {artifact_path}")
    print(f"Final 50 non-tie win rate: {metrics['final_50_non_tie_win_rate']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
