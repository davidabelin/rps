"""Supervised learning pipeline for player-action prediction models.

Role
----
Convert persisted RPS gameplay into train/test datasets, fit supported model
types, and serialize artifacts that can later drive ``active_model`` gameplay
or arena matches.

Cross-Repo Context
------------------
This module is the RPS-side analogue of ``c4_training.supervised``. The two
labs intentionally share the same high-level shape even though the feature
encodings and sample semantics differ.
"""

from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from random import Random
from typing import Any

import numpy as np
from rps_storage.object_store import read_bytes, write_bytes

try:
    from sklearn.metrics import accuracy_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.tree import DecisionTreeClassifier
except Exception as exc:  # pragma: no cover
    accuracy_score = None
    MLPClassifier = None
    DecisionTreeClassifier = None
    SKLEARN_IMPORT_ERROR = str(exc)
else:
    SKLEARN_IMPORT_ERROR = None

SKLEARN_AVAILABLE = DecisionTreeClassifier is not None and MLPClassifier is not None and accuracy_score is not None


@dataclass(slots=True)
class TrainConfig:
    """Configuration for supervised training runs.

    The web training API normalizes request payloads into this structure before
    handing control to the pure training pipeline.
    """

    model_type: str = "decision_tree"
    lookback: int = 5
    test_size: float = 0.2
    learning_rate: float = 0.001
    hidden_layer_sizes: tuple[int, ...] = (64, 32)
    epochs: int = 200
    batch_size: int | str = "auto"
    random_state: int = 42


def training_readiness(rounds: list[dict], lookback: int, minimum_samples: int = 5) -> dict:
    """Summarize whether current round data can support training.

    Parameters
    ----------
    rounds : list[dict]
        Raw round rows from repository.
    lookback : int
        Context window size used to build features.
    minimum_samples : int, default=5
        Minimum feature rows required to allow training.

    Returns
    -------
    dict
        Readiness diagnostics used by training UI.

    Notes
    -----
    This function is intentionally UI-facing: it explains readiness in terms the
    training page can display directly without knowing dataset internals.
    """

    X, _, _ = build_dataset(rounds, lookback=lookback)
    sample_count = int(len(X))
    session_keys = {
        (int(row["game_id"]), int(row["session_index"]))
        for row in rounds
        if "game_id" in row and "session_index" in row
    }
    return {
        "total_round_rows": int(len(rounds)),
        "session_count": int(len(session_keys)),
        "lookback": int(lookback),
        "sample_count": sample_count,
        "minimum_required_samples": int(minimum_samples),
        "sample_formula": "Each game session with n rounds contributes max(0, n - lookback) samples.",
        "can_train": sample_count >= minimum_samples,
        "sklearn_available": bool(SKLEARN_AVAILABLE),
        "sklearn_import_error": SKLEARN_IMPORT_ERROR,
    }


class FrequencyModel:
    """Context-frequency baseline used as a lightweight non-sklearn model.

    Role
    ----
    Provide a deterministic fallback model that preserves the full training and
    activation pipeline even when scikit-learn-backed models are unavailable or
    undesirable.
    """

    def __init__(self, lookback: int) -> None:
        """Initialize context count tables for baseline predictions."""

        self.lookback = lookback
        self.context_counts: dict[tuple[int, ...], np.ndarray] = {}
        self.global_counts = np.ones(3, dtype=float)

    def fit(self, contexts: list[tuple[int, ...]], y: np.ndarray) -> "FrequencyModel":
        """Fit context->label counts with Laplace-style smoothing."""

        for context, label in zip(contexts, y):
            if context not in self.context_counts:
                self.context_counts[context] = np.ones(3, dtype=float)
            self.context_counts[context][int(label)] += 1.0
            self.global_counts[int(label)] += 1.0
        return self

    def predict_context(self, context: tuple[int, ...]) -> int:
        """Predict the most likely label for one context window."""

        counts = self.context_counts.get(context, self.global_counts)
        return int(np.argmax(counts))

    def predict_contexts(self, contexts: list[tuple[int, ...]]) -> np.ndarray:
        """Vectorized prediction over multiple context windows."""

        return np.asarray([self.predict_context(context) for context in contexts], dtype=int)


def _one_hot(action: int) -> list[int]:
    """One-hot encode action id into length-3 feature vector."""

    vec = [0, 0, 0]
    vec[int(action)] = 1
    return vec


def build_dataset(rounds: list[dict], lookback: int) -> tuple[np.ndarray, np.ndarray, list[tuple[int, ...]]]:
    """Construct supervised feature matrix and labels from game rounds.

    Parameters
    ----------
    rounds : list[dict]
        Round rows ordered or unordered across games/sessions.
    lookback : int
        Number of prior rounds used to predict the next player action.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[tuple[int, ...]]]
        ``X`` features, ``y`` labels, and symbolic context tuples used by the
        frequency baseline.

    Role
    ----
    This is the canonical transformation from repository round history into the
    supervised training view of the game.
    """

    if lookback <= 0:
        raise ValueError("lookback must be positive")
    rows = sorted(rounds, key=lambda row: (row["game_id"], row["session_index"], row["round_index"]))
    X: list[list[float]] = []
    y: list[int] = []
    contexts: list[tuple[int, ...]] = []

    current_group: list[dict] = []
    current_key: tuple[int, int] | None = None

    def consume_group(group: list[dict]) -> None:
        """Append feature/label rows for one game-session group."""

        for idx in range(lookback, len(group)):
            window = group[idx - lookback : idx]
            features: list[float] = []
            context: list[int] = []
            for step in window:
                player_action = int(step["player_action"])
                ai_action = int(step["ai_action"])
                reward_delta = int(step["reward_delta"])
                context.append(player_action)
                features.extend(_one_hot(player_action))
                features.extend(_one_hot(ai_action))
                features.append(float(reward_delta))
            X.append(features)
            y.append(int(group[idx]["player_action"]))
            contexts.append(tuple(context))

    for row in rows:
        key = (int(row["game_id"]), int(row["session_index"]))
        if current_key is None:
            current_key = key
        if key != current_key:
            consume_group(current_group)
            current_group = []
            current_key = key
        current_group.append(row)
    if current_group:
        consume_group(current_group)

    if not X:
        return np.asarray([]), np.asarray([]), []
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int), contexts


def _split(
    X: np.ndarray,
    y: np.ndarray,
    contexts: list[tuple[int, ...]],
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[int, ...]], list[tuple[int, ...]]]:
    """Deterministically split dataset into train/test partitions."""

    count = len(X)
    indices = list(range(count))
    rng = Random(random_state)
    rng.shuffle(indices)
    test_count = max(1, int(count * test_size))
    train_count = max(1, count - test_count)
    train_idx = indices[:train_count]
    test_idx = indices[train_count:]
    if not test_idx:
        test_idx = train_idx[-1:]
        train_idx = train_idx[:-1] or train_idx
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    ctx_train = [contexts[idx] for idx in train_idx]
    ctx_test = [contexts[idx] for idx in test_idx]
    return X_train, X_test, y_train, y_test, ctx_train, ctx_test


def _majority_baseline(y_true: np.ndarray, y_train: np.ndarray) -> float:
    """Compute majority-class baseline accuracy for comparison metrics."""

    if len(y_true) == 0:
        return 0.0
    majority = int(np.argmax(np.bincount(y_train if len(y_train) else y_true)))
    return float(np.mean(y_true == majority))


def train_model(rounds: list[dict], config: TrainConfig, artifact_path: str) -> dict[str, Any]:
    """Train one supervised model and persist artifact/metrics.

    Parameters
    ----------
    rounds : list[dict]
        Training round history from repository.
    config : TrainConfig
        Model/training hyperparameters.
    artifact_path : str
        Output destination path (local or ``gs://``).

    Returns
    -------
    dict[str, Any]
        Metrics summary including artifact path.

    Used By
    -------
    ``rps_training.jobs.TrainingJobManager``.

    Side Effects
    ------------
    Serializes the trained artifact to local disk or object storage.
    """

    X, y, contexts = build_dataset(rounds, lookback=config.lookback)
    if len(X) < 5:
        raise RuntimeError("Not enough training samples. Play more rounds before training.")
    X_train, X_test, y_train, y_test, ctx_train, ctx_test = _split(
        X,
        y,
        contexts,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    model_type = config.model_type
    if model_type == "decision_tree":
        if DecisionTreeClassifier is None:
            raise RuntimeError(
                "scikit-learn is required for decision_tree training"
                + (f". Import error: {SKLEARN_IMPORT_ERROR}" if SKLEARN_IMPORT_ERROR else "")
            )
        model = DecisionTreeClassifier(random_state=config.random_state)
        model.fit(X_train, y_train)
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        train_acc = float(accuracy_score(y_train, pred_train)) if accuracy_score else float(np.mean(pred_train == y_train))
        test_acc = float(accuracy_score(y_test, pred_test)) if accuracy_score else float(np.mean(pred_test == y_test))
    elif model_type == "mlp":
        if MLPClassifier is None:
            raise RuntimeError(
                "scikit-learn is required for mlp training"
                + (f". Import error: {SKLEARN_IMPORT_ERROR}" if SKLEARN_IMPORT_ERROR else "")
            )
        model = MLPClassifier(
            hidden_layer_sizes=config.hidden_layer_sizes,
            learning_rate_init=config.learning_rate,
            max_iter=config.epochs,
            batch_size=config.batch_size,
            random_state=config.random_state,
        )
        model.fit(X_train, y_train)
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        train_acc = float(accuracy_score(y_train, pred_train)) if accuracy_score else float(np.mean(pred_train == y_train))
        test_acc = float(accuracy_score(y_test, pred_test)) if accuracy_score else float(np.mean(pred_test == y_test))
    elif model_type == "frequency":
        model = FrequencyModel(lookback=config.lookback).fit(ctx_train, y_train)
        pred_train = model.predict_contexts(ctx_train)
        pred_test = model.predict_contexts(ctx_test)
        train_acc = float(np.mean(pred_train == y_train))
        test_acc = float(np.mean(pred_test == y_test))
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    metrics = {
        "sample_count": int(len(X)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "baseline_accuracy": _majority_baseline(y_test, y_train),
        "lookback": int(config.lookback),
        "epochs": int(config.epochs),
        "batch_size": config.batch_size,
    }
    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "config": asdict(config),
        "model_type": model_type,
        "model": model,
    }
    payload = pickle.dumps(artifact)
    write_bytes(artifact_path, payload, content_type="application/octet-stream")
    metrics["artifact_path"] = artifact_path
    return metrics


def load_artifact(path: str) -> dict[str, Any]:
    """Load serialized model artifact from local or object storage."""

    payload = read_bytes(path)
    return pickle.loads(payload)


def _history_features(history: list[dict], lookback: int) -> tuple[np.ndarray, tuple[int, ...]] | None:
    """Encode most recent history window into model input feature vector."""

    if len(history) < lookback:
        return None
    window = history[-lookback:]
    features: list[float] = []
    context: list[int] = []
    for step in window:
        player_action = int(step["player_action"])
        ai_action = int(step["ai_action"])
        reward_delta = int(step["reward_delta"])
        context.append(player_action)
        features.extend(_one_hot(player_action))
        features.extend(_one_hot(ai_action))
        features.append(float(reward_delta))
    return np.asarray(features, dtype=float), tuple(context)


def predict_player_action(artifact: dict[str, Any], history: list[dict]) -> int | None:
    """Predict the next player action from artifact + recent history.

    Parameters
    ----------
    artifact : dict[str, Any]
        Loaded model artifact.
    history : list[dict]
        Prior transitions in model-history format.

    Returns
    -------
    int | None
        Predicted action id, or ``None`` when history is shorter than lookback.

    Cross-Repo Context
    ------------------
    The special ``active_model`` RPS gameplay path ultimately reaches this logic
    through the model-backed agent implementation.
    """

    config = artifact.get("config", {})
    lookback = int(config.get("lookback", 5))
    encoded = _history_features(history, lookback=lookback)
    if encoded is None:
        return None
    features, context = encoded
    model_type = artifact.get("model_type")
    model = artifact.get("model")
    if model_type == "frequency":
        return int(model.predict_context(context))
    prediction = model.predict(np.asarray([features]))[0]
    return int(prediction)
