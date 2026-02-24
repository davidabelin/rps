"""CLI entry point for supervised model training from stored rounds."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rps_storage import RPSRepository
from rps_storage.object_store import is_gcs_uri, join_storage_path
from rps_training.supervised import TrainConfig, train_model


def main() -> int:
    """Run supervised training CLI and register resulting model artifact."""

    parser = argparse.ArgumentParser(description="Train a supervised RPS model from SQLite round logs.")
    parser.add_argument("--db-path", type=str, default="data/rps.db")
    parser.add_argument("--models-dir", type=str, default="data/models")
    parser.add_argument("--model-type", type=str, default="decision_tree", choices=["decision_tree", "mlp", "frequency"])
    parser.add_argument("--lookback", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-layer-sizes", type=str, default="64,32")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=str, default="auto")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    repository = RPSRepository(args.db_path)
    repository.init_schema()
    rounds = repository.list_rounds_for_training()
    if not rounds:
        raise SystemExit("No rounds available for training. Play games first.")

    hidden_layer_sizes = tuple(int(part.strip()) for part in args.hidden_layer_sizes.split(",") if part.strip())
    config = TrainConfig(
        model_type=args.model_type,
        lookback=args.lookback,
        learning_rate=args.learning_rate,
        hidden_layer_sizes=hidden_layer_sizes,
        epochs=args.epochs,
        batch_size=("auto" if str(args.batch_size).strip().lower() == "auto" else int(args.batch_size)),
        test_size=args.test_size,
        random_state=args.random_state,
    )

    models_dir = str(args.models_dir)
    if not is_gcs_uri(models_dir):
        Path(models_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    artifact_path = join_storage_path(models_dir, f"{config.model_type}_cli_{timestamp}.pkl")
    metrics = train_model(rounds=rounds, config=config, artifact_path=artifact_path)
    model_row = repository.create_model(
        name=f"{config.model_type}-cli-{timestamp}",
        model_type=config.model_type,
        artifact_path=artifact_path,
        lookback=config.lookback,
        metrics=metrics,
    )
    print("Training completed")
    print(f"Model ID: {model_row['id']}")
    print(f"Artifact: {artifact_path}")
    print(f"Metrics: {metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
