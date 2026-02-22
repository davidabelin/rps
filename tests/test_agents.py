from rps_agents.heuristic.basic import CopyOpponentAgent
from rps_agents.heuristic.constants import HitLastOwnActionAgent
from rps_agents.heuristic.ensemble import RotatingEnsembleAgent
from rps_agents.heuristic.memory_patterns import MemoryPatternAgent
from rps_core.types import RoundObservation, RoundTransition


def test_copy_opponent_follows_last_action():
    agent = CopyOpponentAgent()
    agent.reset(seed=11)
    first = agent.select_action(RoundObservation(step=0, last_opponent_action=None, cumulative_reward=0))
    assert first in (0, 1, 2)
    transition = RoundTransition(
        observation=RoundObservation(step=0, last_opponent_action=None, cumulative_reward=0),
        action=first,
        opponent_action=2,
        outcome="ai",
        reward_delta=-1,
        round_index=0,
    )
    agent.observe(transition)
    second = agent.select_action(RoundObservation(step=1, last_opponent_action=2, cumulative_reward=-1))
    assert second == 2


def test_hit_last_own_action_cycles():
    agent = HitLastOwnActionAgent()
    agent.reset(seed=3)
    assert agent.select_action(RoundObservation(step=0, last_opponent_action=None, cumulative_reward=0)) == 1
    assert agent.select_action(RoundObservation(step=1, last_opponent_action=0, cumulative_reward=0)) == 2
    assert agent.select_action(RoundObservation(step=2, last_opponent_action=1, cumulative_reward=0)) == 0


def test_memory_pattern_agent_returns_valid_action():
    agent = MemoryPatternAgent(memory_length=4)
    agent.reset(seed=7)
    for step in range(10):
        action = agent.select_action(RoundObservation(step=step, last_opponent_action=step % 3, cumulative_reward=0))
        assert action in (0, 1, 2)
        agent.observe(
            RoundTransition(
                observation=RoundObservation(step=step, last_opponent_action=(step - 1) % 3, cumulative_reward=0),
                action=action,
                opponent_action=step % 3,
                outcome="tie",
                reward_delta=0,
                round_index=step,
            )
        )


def test_rotating_ensemble_stays_stable_over_long_sequence():
    agent = RotatingEnsembleAgent()
    agent.reset(seed=19)
    cumulative_reward = 0
    last_opponent_action = None
    for step in range(360):
        obs = RoundObservation(step=step, last_opponent_action=last_opponent_action, cumulative_reward=cumulative_reward)
        action = agent.select_action(obs)
        assert action in (0, 1, 2)
        opponent_action = (step + 1) % 3
        reward_delta = 1 if action == (opponent_action + 1) % 3 else -1 if opponent_action == (action + 1) % 3 else 0
        cumulative_reward += reward_delta
        last_opponent_action = opponent_action
        agent.observe(
            RoundTransition(
                observation=obs,
                action=action,
                opponent_action=opponent_action,
                outcome="player" if reward_delta > 0 else "ai" if reward_delta < 0 else "tie",
                reward_delta=reward_delta,
                round_index=step,
            )
        )
