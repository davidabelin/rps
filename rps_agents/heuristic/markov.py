from __future__ import annotations

from collections import defaultdict

import numpy as np

from rps_agents.heuristic.common import RNGMixin
from rps_core.types import RoundObservation, RoundTransition


class MarkovAgent(RNGMixin):
    name = "markov"

    def __init__(
        self,
        order: int = 2,
        refresh_interval: int = 250,
        deterministic_horizon: int = 500,
        mirror_horizon: int = 900,
    ) -> None:
        super().__init__()
        self.order = order
        self.refresh_interval = refresh_interval
        self.deterministic_horizon = deterministic_horizon
        self.mirror_horizon = mirror_horizon
        self.table = defaultdict(lambda: [1, 1, 1])
        self.action_seq: list[int] = []

    def reset(self, seed: int | None) -> None:
        super().reset(seed)
        self.table = defaultdict(lambda: [1, 1, 1])
        self.action_seq = []

    def select_action(self, obs: RoundObservation) -> int:
        if obs.step % self.refresh_interval == 0:
            self.table = defaultdict(lambda: [1, 1, 1])
            self.action_seq = []

        if len(self.action_seq) <= 2 * self.order + 1:
            action = int(np.random.randint(3))
            if obs.step > 0 and obs.last_opponent_action is not None:
                self.action_seq.extend([obs.last_opponent_action, action])
            else:
                self.action_seq.append(action)
            return action

        if obs.last_opponent_action is None:
            return self._rand_action()

        key = "".join(str(a) for a in self.action_seq[:-1])
        self.table[key][obs.last_opponent_action] += 1

        self.action_seq[:-2] = self.action_seq[2:]
        self.action_seq[-2] = obs.last_opponent_action

        key = "".join(str(a) for a in self.action_seq[:-1])
        if obs.step < self.deterministic_horizon:
            next_opp = int(np.argmax(self.table[key]))
        else:
            scores = np.asarray(self.table[key], dtype=float)
            next_opp = int(np.random.choice(3, p=scores / scores.sum()))

        action = (next_opp + 1) % 3
        if obs.step > self.mirror_horizon:
            action = next_opp
        self.action_seq[-1] = action
        return int(action)

    def observe(self, transition: RoundTransition) -> None:
        return None
