from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from rps_agents.base import AgentProtocol
from rps_benchmarks.canonical import CANONICAL_BOT_FACTORIES, action_to_symbol, symbol_to_action
from rps_core.scoring import score_round
from rps_core.types import RoundObservation, RoundTransition


@dataclass(slots=True)
class BenchStats:
    rounds: int = 0
    wins: int = 0
    losses: int = 0
    ties: int = 0

    def record(self, reward: int) -> None:
        self.rounds += 1
        if reward > 0:
            self.wins += 1
        elif reward < 0:
            self.losses += 1
        else:
            self.ties += 1

    def as_dict(self) -> dict:
        non_ties = self.wins + self.losses
        return {
            "rounds": self.rounds,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "win_rate": self.wins / self.rounds if self.rounds else 0.0,
            "non_tie_win_rate": self.wins / non_ties if non_ties else 0.0,
        }


def _play_against_bot(agent_factory: Callable[[], AgentProtocol], bot_name: str, rounds: int, seed: int | None) -> dict:
    bot_factory = CANONICAL_BOT_FACTORIES[bot_name]
    if bot_name == "random":
        bot = bot_factory(seed=seed or 7)
    else:
        bot = bot_factory()
    bot.reset()
    agent = agent_factory()
    agent.reset(seed=seed)

    stats = BenchStats()
    agent_cumulative_reward = 0
    last_bot_action: int | None = None
    prev_agent_symbol = ""

    for step in range(rounds):
        observation = RoundObservation(
            step=step,
            last_opponent_action=last_bot_action,
            cumulative_reward=agent_cumulative_reward,
        )
        agent_action = int(agent.select_action(observation))
        bot_symbol = bot.play(prev_agent_symbol)
        bot_action = symbol_to_action(bot_symbol)
        reward = score_round(agent_action, bot_action)
        stats.record(reward)
        agent.observe(
            RoundTransition(
                observation=observation,
                action=agent_action,
                opponent_action=bot_action,
                outcome="player" if reward > 0 else "ai" if reward < 0 else "tie",
                reward_delta=reward,
                round_index=step,
            )
        )
        agent_cumulative_reward += reward
        last_bot_action = bot_action
        prev_agent_symbol = action_to_symbol(agent_action)

    payload = stats.as_dict()
    payload["bot"] = bot_name
    return payload


def benchmark_agent(agent_factory: Callable[[], AgentProtocol], rounds: int = 1000, seed: int = 7) -> dict:
    bots = ["quincy", "abbey", "kris", "mrugesh"]
    results = []
    for offset, bot in enumerate(bots):
        results.append(_play_against_bot(agent_factory, bot_name=bot, rounds=rounds, seed=seed + offset))
    overall_non_tie = sum(item["wins"] for item in results) / max(
        1, sum(item["wins"] + item["losses"] for item in results)
    )
    overall_win = sum(item["wins"] for item in results) / max(1, sum(item["rounds"] for item in results))
    return {
        "rounds_per_bot": rounds,
        "overall_win_rate": overall_win,
        "overall_non_tie_win_rate": overall_non_tie,
        "target_non_tie_win_rate": 0.60,
        "meets_target": overall_non_tie >= 0.60,
        "results": results,
    }
