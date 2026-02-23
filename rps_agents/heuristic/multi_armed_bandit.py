from __future__ import annotations

"""Thompson-sampling ensemble over multiple heuristic predictors."""

from collections import Counter
from dataclasses import dataclass

import numpy as np

from rps_agents.heuristic.common import RNGMixin
from rps_core.scoring import score_round
from rps_core.types import RoundObservation, RoundTransition


def _sample_transition(
    history: list[dict],
    key: str,
    deterministic: bool,
    decay: float,
    init_value: float,
    max_history: int,
) -> int:
    """Sample predicted next action from decayed transition counts.

    Parameters
    ----------
    history : list[dict]
        Sequence of previous move records.
    key : str
        Which action stream to model (``"opponent"`` or ``"action"``).
    deterministic : bool
        If ``True``, return argmax instead of stochastic sample.
    decay : float
        Exponential decay factor for old observations.
    init_value : float
        Prior count value for each transition bucket.
    max_history : int
        Maximum number of recent records used for runtime cost control.
    """

    if len(history) > max_history:
        history = history[-max_history:]
    matrix = np.zeros((3, 3), dtype=float) + init_value
    for i in range(len(history) - 1):
        matrix = (matrix - init_value) / decay + init_value
        left = int(history[i][key])
        right = int(history[i + 1][key])
        matrix[left, right] += 1.0
    previous = int(history[-1][key])
    row = matrix[previous]
    if deterministic:
        return int(np.argmax(row))
    return int(np.random.choice([0, 1, 2], p=row / row.sum()))


@dataclass
class Candidate:
    """Beta posterior parameters for one arm/predictor."""

    alpha: float = 1.0
    beta: float = 1.0


class MultiArmedBanditAgent(RNGMixin):
    """Choose among predictor arms with Thompson sampling.

    Notes
    -----
    Each arm proposes an action; arm posteriors are updated from realized round
    outcomes against the opponent.
    """

    name = "multi_armed_bandit"

    def __init__(self, step_size: float = 2.0, decay_rate: float = 1.05, max_transition_history: int = 180) -> None:
        """Initialize candidate arms and posterior state."""

        super().__init__()
        self.step_size = step_size
        self.decay_rate = decay_rate
        self.max_transition_history = max(30, int(max_transition_history))
        self.history: list[dict] = []
        self.bandits: dict[str, Candidate] = {
            "mirror_0": Candidate(),
            "mirror_1": Candidate(),
            "mirror_2": Candidate(),
            "self_0": Candidate(),
            "self_1": Candidate(),
            "self_2": Candidate(),
            "popular_beater": Candidate(),
            "anti_popular_beater": Candidate(),
            "transition_random": Candidate(),
            "transition_deterministic": Candidate(),
            "transition_self": Candidate(),
            "transition_self_det": Candidate(),
        }
        self._last_predictions: dict[str, int] = {}
        self._selected_agent = "mirror_0"

    def reset(self, seed: int | None) -> None:
        """Reset RNG, history, and all arm posteriors."""

        super().reset(seed)
        self.history = []
        self._last_predictions = {}
        self._selected_agent = "mirror_0"
        self.bandits = {name: Candidate() for name in self.bandits}

    def _predict(self, name: str) -> int:
        """Return one arm's proposed action for the current history."""

        if not self.history:
            return self._rand_action()
        last = self.history[-1]
        if name.startswith("mirror_"):
            shift = int(name.split("_")[1])
            return (int(last["opponent"]) + shift) % 3
        if name.startswith("self_"):
            shift = int(name.split("_")[1])
            return (int(last["action"]) + shift) % 3
        if name == "popular_beater":
            counts = Counter(item["opponent"] for item in self.history)
            return (counts.most_common(1)[0][0] + 1) % 3
        if name == "anti_popular_beater":
            counts = Counter(item["action"] for item in self.history)
            return (counts.most_common(1)[0][0] + 2) % 3
        if name == "transition_random":
            pred = _sample_transition(
                self.history,
                "opponent",
                deterministic=False,
                decay=1.0,
                init_value=0.1,
                max_history=self.max_transition_history,
            )
            return (pred + 1) % 3
        if name == "transition_deterministic":
            pred = _sample_transition(
                self.history,
                "opponent",
                deterministic=True,
                decay=1.0,
                init_value=0.1,
                max_history=self.max_transition_history,
            )
            return (pred + 1) % 3
        if name == "transition_self":
            pred = _sample_transition(
                self.history,
                "action",
                deterministic=False,
                decay=1.05,
                init_value=0.1,
                max_history=self.max_transition_history,
            )
            return (pred + 2) % 3
        if name == "transition_self_det":
            pred = _sample_transition(
                self.history,
                "action",
                deterministic=True,
                decay=1.05,
                init_value=0.1,
                max_history=self.max_transition_history,
            )
            return (pred + 2) % 3
        return self._rand_action()

    def select_action(self, obs: RoundObservation) -> int:
        """Sample arm scores and return action from highest sampled arm."""

        best_name = None
        best_value = -1.0
        self._last_predictions = {}
        for name, state in self.bandits.items():
            sampled = np.random.beta(state.alpha, state.beta)
            prediction = self._predict(name)
            self._last_predictions[name] = prediction
            if sampled > best_value:
                best_value = sampled
                best_name = name
        self._selected_agent = best_name or "mirror_0"
        return int(self._last_predictions[self._selected_agent])

    def observe(self, transition: RoundTransition) -> None:
        """Update all arm posteriors from realized opponent action."""

        opponent_action = int(transition.opponent_action)
        for name, predicted in self._last_predictions.items():
            state = self.bandits[name]
            state.alpha = (state.alpha - 1.0) / self.decay_rate + 1.0
            state.beta = (state.beta - 1.0) / self.decay_rate + 1.0
            result = score_round(predicted, opponent_action)
            if result > 0:
                state.alpha += self.step_size
            elif result < 0:
                state.beta += self.step_size
            else:
                state.alpha += self.step_size / 2
                state.beta += self.step_size / 2
        self.history.append({"action": int(transition.action), "opponent": opponent_action, "agent": self._selected_agent})
