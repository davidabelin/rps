"""Online decision-tree heuristic using rolling local/global features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rps_agents.heuristic.common import RNGMixin
from rps_core.types import RoundObservation, RoundTransition

try:
    from sklearn.tree import DecisionTreeClassifier
except Exception:  # pragma: no cover
    DecisionTreeClassifier = None


@dataclass
class _Move:
    """Internal move record structure (currently reserved/auxiliary)."""

    step: int
    action: int
    opp_action: int


def _construct_local_features(rollouts: dict[str, list[int]]) -> np.ndarray:
    """Build short-horizon handcrafted feature vector."""

    features = np.array([[step % k for step in rollouts["steps"]] for k in (2, 3, 5)], dtype=float)
    features = np.append(features, rollouts["steps"])
    features = np.append(features, rollouts["actions"])
    features = np.append(features, rollouts["opp-actions"])
    return features


def _construct_global_features(rollouts: dict[str, list[int]]) -> np.ndarray:
    """Build long-horizon aggregate frequency feature vector."""

    features: list[float] = []
    for key in ("actions", "opp-actions"):
        for choice in range(3):
            features.append(float(np.mean([item == choice for item in rollouts[key]])))
    return np.array(features, dtype=float)


def _construct_features(short_stats: dict[str, list[int]], long_stats: dict[str, list[int]]) -> np.ndarray:
    """Concatenate local + global feature blocks."""

    return np.concatenate([_construct_local_features(short_stats), _construct_global_features(long_stats)])


class DecisionTreeHeuristicAgent(RNGMixin):
    """Train a lightweight decision tree online during a single game.

    Notes
    -----
    This is a heuristic model, not persisted supervised training. It updates
    from transitions seen during one game session.
    """

    name = "decision_tree"

    def __init__(
        self,
        window: int = 5,
        min_samples: int = 25,
        random_state: int = 42,
        retrain_interval: int = 6,
        max_history: int = 180,
    ) -> None:
        """Configure rolling window, retrain cadence, and history cap."""

        super().__init__()
        self.window = window
        self.min_samples = min_samples
        self.random_state = random_state
        self.retrain_interval = max(1, int(retrain_interval))
        self.max_history = max(int(max_history), int(min_samples + window + 2))
        self.rollouts_hist: dict[str, list[int]] = {"steps": [], "actions": [], "opp-actions": []}
        self.classifier = DecisionTreeClassifier(random_state=random_state) if DecisionTreeClassifier else None
        self._last_fit_step = -1
        self._is_fitted = False

    def reset(self, seed: int | None) -> None:
        """Reset RNG and clear per-session rollout history."""

        super().reset(seed)
        self.rollouts_hist = {"steps": [], "actions": [], "opp-actions": []}
        self._last_fit_step = -1
        self._is_fitted = False

    def _update_rollouts(self, transition: RoundTransition) -> None:
        """Append one transition into rollout buffers."""

        self.rollouts_hist["steps"].append(transition.round_index)
        self.rollouts_hist["actions"].append(transition.action)
        self.rollouts_hist["opp-actions"].append(transition.opponent_action)

    def _recent_rollouts(self) -> dict[str, list[int]]:
        """Return recent rollout slices bounded by ``max_history``."""

        total = len(self.rollouts_hist["steps"])
        start = max(0, total - self.max_history)
        return {key: values[start:] for key, values in self.rollouts_hist.items()}

    def _fit_if_due(self, obs: RoundObservation) -> None:
        """Refit classifier only when cadence/threshold conditions are met."""

        if self.classifier is None:
            return
        total_steps = len(self.rollouts_hist["steps"])
        if obs.step <= self.min_samples + self.window:
            return
        if total_steps < max(self.min_samples, self.window + 1):
            return
        should_refit = (not self._is_fitted) or ((obs.step - self._last_fit_step) >= self.retrain_interval)
        if not should_refit:
            return

        capped = self._recent_rollouts()
        capped_steps = len(capped["steps"])
        if capped_steps < self.window + 2:
            return

        feature_rows: list[np.ndarray] = []
        for i in range(capped_steps - self.window + 1):
            short_stats = {key: capped[key][i : i + self.window] for key in capped}
            long_stats = {key: capped[key][: i + self.window] for key in capped}
            feature_rows.append(_construct_features(short_stats, long_stats))
        if len(feature_rows) < 2:
            return

        train_x = np.asarray(feature_rows[:-1], dtype=float)
        train_y = np.asarray(capped["opp-actions"][self.window :], dtype=int)
        if len(train_x) == 0 or len(train_y) == 0 or len(train_x) != len(train_y):
            return
        try:
            self.classifier.fit(train_x, train_y)
            self._last_fit_step = int(obs.step)
            self._is_fitted = True
        except ValueError:
            return

    def _predict_next_opponent_action(self) -> int | None:
        """Predict next opponent action with currently fitted classifier."""

        if self.classifier is None or not self._is_fitted:
            return None
        capped = self._recent_rollouts()
        capped_steps = len(capped["steps"])
        if capped_steps < self.window:
            return None
        short_stats = {key: capped[key][-self.window :] for key in capped}
        long_stats = capped
        sample = _construct_features(short_stats, long_stats).reshape(1, -1)
        try:
            return int(self.classifier.predict(sample)[0])
        except ValueError:
            return None

    def select_action(self, obs: RoundObservation) -> int:
        """Predict opponent move and play counter action."""

        if self.classifier is None:
            return self._rand_action()
        if obs.step <= self.min_samples + self.window:
            return self._rand_action()
        total_steps = len(self.rollouts_hist["steps"])
        if total_steps < max(self.min_samples, self.window + 1):
            return self._rand_action()

        self._fit_if_due(obs)
        next_opp_action_pred = self._predict_next_opponent_action()
        if next_opp_action_pred is None:
            return self._rand_action()
        return int((next_opp_action_pred + 1) % 3)

    def observe(self, transition: RoundTransition) -> None:
        """Store current transition for future online fitting."""

        self._update_rollouts(transition)
