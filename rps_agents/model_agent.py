from __future__ import annotations

from random import Random

from rps_core.scoring import counter_action
from rps_core.types import RoundObservation, RoundTransition
from rps_training.supervised import load_artifact, predict_player_action


class ModelBackedAgent:
    name = "active_model"

    def __init__(self, artifact_path: str) -> None:
        self._artifact_path = artifact_path
        self._artifact = load_artifact(artifact_path)
        self._history: list[dict] = []
        self._rng = Random()
        self._policy = self._artifact.get("policy")
        self._q_table = self._artifact.get("q_table")
        self._model_type = str(self._artifact.get("model_type", "decision_tree"))

    def reset(self, seed: int | None) -> None:
        self._rng.seed(seed)
        self._history = []

    @staticmethod
    def _state_index(last_opponent_action: int | None) -> int:
        if last_opponent_action is None:
            return 3
        return int(last_opponent_action)

    def select_action(self, obs: RoundObservation) -> int:
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
        self._history.append(
            {
                "player_action": int(transition.opponent_action),
                "ai_action": int(transition.action),
                "reward_delta": -int(transition.reward_delta),
            }
        )
