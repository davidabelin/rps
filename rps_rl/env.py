"""Minimal RL environment primitives used by current tabular trainer."""

from __future__ import annotations

from dataclasses import dataclass

from rps_core.scoring import score_round


@dataclass
class RLStepResult:
    """Result payload returned by one environment step."""

    reward: int
    done: bool
    info: dict


class SimpleRPSEnv:
    """Placeholder RL environment to support phase-2 expansion."""

    action_space = [0, 1, 2]

    def __init__(self) -> None:
        """Initialize environment state."""

        self.step_count = 0
        self.last_opponent_action: int | None = None

    def reset(self) -> dict:
        """Reset to initial state and return first observation."""

        self.step_count = 0
        self.last_opponent_action = None
        return {"last_opponent_action": self.last_opponent_action}

    def step(self, action: int, opponent_action: int) -> tuple[dict, RLStepResult]:
        """Advance environment by one action pair.

        Parameters
        ----------
        action : int
            Agent action id.
        opponent_action : int
            Opponent action id.

        Returns
        -------
        tuple[dict, RLStepResult]
            Next observation and transition result.
        """

        reward = score_round(action, opponent_action)
        self.step_count += 1
        self.last_opponent_action = opponent_action
        observation = {"last_opponent_action": self.last_opponent_action}
        return observation, RLStepResult(reward=reward, done=False, info={"step": self.step_count})
