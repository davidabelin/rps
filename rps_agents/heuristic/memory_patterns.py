from __future__ import annotations

"""Pattern-memory heuristic that maps short histories to likely responses."""

from dataclasses import dataclass, field

from rps_agents.heuristic.common import RNGMixin
from rps_core.types import RoundObservation, RoundTransition


@dataclass
class PatternRecord:
    """Stored memory pattern and distribution of following opponent actions."""

    actions: list[int]
    opp_next_actions: dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0, 2: 0})


class MemoryPatternAgent(RNGMixin):
    """Predict from repeated mixed-action patterns in recent history."""

    name = "memory_patterns"

    def __init__(self, memory_length: int = 6) -> None:
        """Initialize memory length and pattern table."""

        super().__init__()
        self.memory_length = memory_length
        self.current_memory: list[int] = []
        self.patterns: list[PatternRecord] = []

    def reset(self, seed: int | None) -> None:
        """Reset RNG and clear pattern memory."""

        super().reset(seed)
        self.current_memory = []
        self.patterns = []

    def _find_pattern(self, memory: list[int]) -> PatternRecord | None:
        """Find exact prefix-match pattern record for given memory window."""

        for pattern in self.patterns:
            if pattern.actions[: self.memory_length] == memory[: self.memory_length]:
                return pattern
        return None

    def select_action(self, obs: RoundObservation) -> int:
        """Choose action using memorized pattern continuation frequencies."""

        if len(self.current_memory) > self.memory_length:
            previous_memory = self.current_memory[: self.memory_length]
            previous_pattern = self._find_pattern(previous_memory)
            if previous_pattern is None:
                previous_pattern = PatternRecord(actions=previous_memory.copy())
                self.patterns.append(previous_pattern)
            if obs.last_opponent_action is not None:
                previous_pattern.opp_next_actions[int(obs.last_opponent_action)] += 1
            del self.current_memory[:2]

        action = self._rand_action()
        pattern = self._find_pattern(self.current_memory)
        if pattern is not None:
            action = max(pattern.opp_next_actions, key=pattern.opp_next_actions.get)
            action = (action + 1) % 3
        self.current_memory.append(action)
        return int(action)

    def observe(self, transition: RoundTransition) -> None:
        """Append observed opponent action into mixed memory stream."""

        self.current_memory.append(int(transition.opponent_action))
