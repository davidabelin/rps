"""Benchmark harness for evaluating agents against canonical bots."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

from rps_agents.base import AgentProtocol
from rps_benchmarks.canonical import (
    CANONICAL_BOT_FACTORIES,
    action_to_symbol,
    get_benchmark_suite,
    symbol_to_action,
)
from rps_core.scoring import score_round
from rps_core.types import RoundObservation, RoundTransition


@dataclass(slots=True)
class BenchStats:
    """Mutable benchmark counters for one matchup."""

    rounds: int = 0
    wins: int = 0
    losses: int = 0
    ties: int = 0

    def record(self, reward: int) -> None:
        """Accumulate one round outcome from reward sign."""

        self.rounds += 1
        if reward > 0:
            self.wins += 1
        elif reward < 0:
            self.losses += 1
        else:
            self.ties += 1

    def as_dict(self) -> dict:
        """Return summary metrics dictionary for JSON response."""

        non_ties = self.wins + self.losses
        return {
            "rounds": self.rounds,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "win_rate": self.wins / self.rounds if self.rounds else 0.0,
            "non_tie_win_rate": self.wins / non_ties if non_ties else 0.0,
        }


def _play_against_bot(
    agent_factory: Callable[[], AgentProtocol],
    bot_name: str,
    rounds: int,
    seed: int | None,
    *,
    started: float | None = None,
    max_elapsed_seconds: float | None = None,
) -> dict:
    """Run one agent-vs-canonical-bot matchup.

    Parameters
    ----------
    agent_factory : Callable[[], AgentProtocol]
        Factory creating a fresh agent instance.
    bot_name : str
        Canonical bot identifier.
    rounds : int
        Number of rounds to play.
    seed : int | None
        Optional seed used for deterministic setup where applicable.
    """

    bot_factory = CANONICAL_BOT_FACTORIES[bot_name]
    if bot_name in {"random", "nash_equilibrium"}:
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
        if (
            max_elapsed_seconds is not None
            and started is not None
            and step % 50 == 0
            and (perf_counter() - started) >= max_elapsed_seconds
        ):
            raise TimeoutError(
                f"Benchmark exceeded time budget after {bot_name} step {step}. "
                f"Try fewer rounds, core suite, or CLI benchmark script."
            )
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


def _resolve_bots(suite: str, bots: list[str] | None) -> tuple[str, list[str]]:
    """Resolve requested benchmark opponent set."""

    if bots:
        normalized = [str(item).strip().lower() for item in bots if str(item).strip()]
        if not normalized:
            raise ValueError("Custom bot list is empty.")
        unknown = [item for item in normalized if item not in CANONICAL_BOT_FACTORIES]
        if unknown:
            raise ValueError(f"Unknown benchmark bot(s): {', '.join(unknown)}")
        return "custom", normalized
    return suite, list(get_benchmark_suite(suite))


def benchmark_agent(
    agent_factory: Callable[[], AgentProtocol],
    rounds: int = 1000,
    seed: int = 7,
    suite: str = "core",
    bots: list[str] | None = None,
    max_elapsed_seconds: float | None = None,
) -> dict:
    """Evaluate one agent against standard canonical bot set.

    Parameters
    ----------
    agent_factory : Callable[[], AgentProtocol]
        Factory creating a fresh evaluated agent.
    rounds : int, default=1000
        Rounds played against each canonical bot.
    seed : int, default=7
        Base seed; per-bot offsets are applied for variance.
    suite : str, default="core"
        Benchmark suite name. Ignored when ``bots`` is provided.
    bots : list[str] | None, default=None
        Optional explicit list of benchmark bot names.
    max_elapsed_seconds : float | None, default=None
        Optional wall-clock time budget in seconds for the full benchmark run.

    Returns
    -------
    dict
        Aggregate and per-bot benchmark metrics.
    """

    started = perf_counter()
    resolved_suite, resolved_bots = _resolve_bots(suite=suite, bots=bots)
    results = []
    for offset, bot in enumerate(resolved_bots):
        if max_elapsed_seconds is not None and (perf_counter() - started) >= max_elapsed_seconds:
            raise TimeoutError(
                "Benchmark exceeded time budget before completion. "
                "Try fewer rounds, core suite, or CLI benchmark script."
            )
        results.append(
            _play_against_bot(
                agent_factory,
                bot_name=bot,
                rounds=rounds,
                seed=seed + offset,
                started=started,
                max_elapsed_seconds=max_elapsed_seconds,
            )
        )
    overall_non_tie = sum(item["wins"] for item in results) / max(
        1, sum(item["wins"] + item["losses"] for item in results)
    )
    overall_win = sum(item["wins"] for item in results) / max(1, sum(item["rounds"] for item in results))
    target_by_suite = {
        "core": 0.60,
        "extended": 0.55,
    }
    target = target_by_suite.get(resolved_suite)
    return {
        "suite": resolved_suite,
        "bots": resolved_bots,
        "rounds_per_bot": rounds,
        "overall_win_rate": overall_win,
        "overall_non_tie_win_rate": overall_non_tie,
        "target_non_tie_win_rate": target,
        "meets_target": (overall_non_tie >= target) if isinstance(target, float) else None,
        "results": results,
    }
