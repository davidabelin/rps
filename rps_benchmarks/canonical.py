"""Canonical benchmark bot implementations and symbol/action adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random


def _counter(action: str) -> str:
    """Return symbol that beats the input symbol."""

    mapping = {"R": "P", "P": "S", "S": "R"}
    return mapping[action]


@dataclass
class QuincyBot:
    """Periodic sequence bot mirrored from common RPS benchmark set."""

    counter: int = 0
    choices: list[str] = field(default_factory=lambda: ["R", "R", "P", "P", "S"])

    def reset(self) -> None:
        """Reset internal sequence counter."""

        self.counter = 0

    def play(self, _prev: str) -> str:
        """Return next sequence symbol."""

        self.counter += 1
        return self.choices[self.counter % len(self.choices)]


@dataclass
class KrisBot:
    """Always counter opponent's previous symbol."""

    def reset(self) -> None:
        """No-op: stateless bot."""

        return None

    def play(self, prev_opponent_play: str) -> str:
        """Return counter of previous opponent symbol."""

        if not prev_opponent_play:
            prev_opponent_play = "R"
        return _counter(prev_opponent_play)


@dataclass
class MrugeshBot:
    """Counter the mode of opponent's recent history window."""

    opponent_history: list[str] = field(default_factory=list)

    def reset(self) -> None:
        """Clear observed opponent history."""

        self.opponent_history = []

    def play(self, prev_opponent_play: str) -> str:
        """Return counter to most frequent recent opponent symbol."""

        self.opponent_history.append(prev_opponent_play)
        last_ten = self.opponent_history[-10:]
        non_empty = [item for item in last_ten if item]
        if not non_empty:
            most_frequent = "S"
        else:
            most_frequent = max(set(non_empty), key=non_empty.count)
        return _counter(most_frequent)


@dataclass
class AbbeyBot:
    """Second-order sequence-frequency predictor from canonical benchmark."""

    opponent_history: list[str] = field(default_factory=list)
    play_order: dict[str, int] = field(
        default_factory=lambda: {
            "RR": 0,
            "RP": 0,
            "RS": 0,
            "PR": 0,
            "PP": 0,
            "PS": 0,
            "SR": 0,
            "SP": 0,
            "SS": 0,
        }
    )

    def reset(self) -> None:
        """Reset sequence tables and history."""

        self.opponent_history = []
        self.play_order = {key: 0 for key in self.play_order.keys()}

    def play(self, prev_opponent_play: str) -> str:
        """Predict from two-symbol sequence counts and counter prediction."""

        if not prev_opponent_play:
            prev_opponent_play = "R"
        self.opponent_history.append(prev_opponent_play)

        last_two = "".join(self.opponent_history[-2:])
        if len(last_two) == 2:
            self.play_order[last_two] += 1

        potential = [prev_opponent_play + "R", prev_opponent_play + "P", prev_opponent_play + "S"]
        sub_order = {key: self.play_order[key] for key in potential}
        prediction = max(sub_order, key=sub_order.get)[-1:]
        return _counter(prediction)


class RandomBot:
    """Uniform random canonical bot."""

    def __init__(self, seed: int = 7) -> None:
        """Initialize RNG with optional seed."""

        self._rng = Random(seed)

    def reset(self) -> None:
        """No-op: RNG state is intentionally persistent."""

        return None

    def play(self, _prev: str) -> str:
        """Sample one random symbol."""

        return self._rng.choice(["R", "P", "S"])


CANONICAL_BOT_FACTORIES = {
    "quincy": QuincyBot,
    "abbey": AbbeyBot,
    "kris": KrisBot,
    "mrugesh": MrugeshBot,
    "random": RandomBot,
}


def action_to_symbol(action: int) -> str:
    """Convert integer action id into symbol representation."""

    return {0: "R", 1: "P", 2: "S"}[int(action)]


def symbol_to_action(symbol: str) -> int:
    """Convert symbol representation into integer action id."""

    return {"R": 0, "P": 1, "S": 2}[symbol]
