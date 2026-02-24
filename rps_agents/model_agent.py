"""Model-backed agent wrapper used for ``active_model`` gameplay."""

from __future__ import annotations

from random import Random

from rps_core.scoring import counter_action
from rps_core.types import RoundObservation, RoundTransition
from rps_training.supervised import load_artifact, predict_player_action


class ModelBackedAgent:
    """Serve a trained artifact through the standard agent protocol.

    Supported artifact families:

    - supervised artifacts (decision tree / MLP / frequency baseline)
    - tabular RL artifact (``rl_qtable``)
    """

    name = "active_model"

    def __init__(self, artifact_path: str) -> None:
        """Load artifact metadata/model state from storage."""

        self._artifact_path = artifact_path
        self._artifact = load_artifact(artifact_path)
        self._history: list[dict] = []
        self._rng = Random()
        self._policy = self._artifact.get("policy")
        self._q_table = self._artifact.get("q_table")
        self._model_type = str(self._artifact.get("model_type", "decision_tree"))
        config = self._artifact.get("config", {}) if isinstance(self._artifact, dict) else {}
        lookback = int(config.get("lookback", 5)) if isinstance(config, dict) else 5
        self._history_cap = max(lookback + 4, 16)

    def reset(self, seed: int | None) -> None:
        """Reset per-session history and RNG seed."""

        self._rng.seed(seed)
        self._history = []

    @staticmethod
    def _state_index(last_opponent_action: int | None) -> int:
        """Map previous opponent action into Q-table state index."""

        if last_opponent_action is None:
            return 3
        return int(last_opponent_action)

    def select_action(self, obs: RoundObservation) -> int:
        """Select the next action from the loaded artifact policy.

        Parameters
        ----------
        obs : RoundObservation
            Current runtime observation.

        Returns
        -------
        int
            Action code ``0..2``.
        """

        if self._model_type == "rl_qtable":
            state = self._state_index(obs.last_opponent_action)
            if self._policy and state < len(self._policy):
                return int(self._policy[state])
            if self._q_table and state < len(self._q_table):
                row = self._q_table[state]
                return int(max(range(len(row)), key=lambda idx: row[idx]))
            return self._rng.randrange(0, 3)
        prediction = predict_player_action(self._artifact, self._history)
        if prediction is None:
            return self._rng.randrange(0, 3)
        return counter_action(prediction)

    def observe(self, transition: RoundTransition) -> None:
        """Append transition in supervised-model history feature format."""

        self._history.append(
            {
                "player_action": int(transition.opponent_action),
                "ai_action": int(transition.action),
                "reward_delta": -int(transition.reward_delta),
            }
        )
        if len(self._history) > self._history_cap:
            self._history = self._history[-self._history_cap :]
