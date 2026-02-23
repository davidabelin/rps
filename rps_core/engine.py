from __future__ import annotations

"""Round execution helpers for human-vs-agent play loops."""

from rps_agents.base import AgentProtocol
from rps_core.scoring import reward_to_outcome, score_round
from rps_core.types import RoundObservation, RoundResult, RoundTransition, normalize_action


def hydrate_agent(agent: AgentProtocol, rounds: list[dict]) -> RoundObservation:
    """Backward-compatible alias for :func:`replay_observation`.

    Parameters
    ----------
    agent : AgentProtocol
        Agent instance to hydrate from history.
    rounds : list[dict]
        Stored rounds for one game session.

    Returns
    -------
    RoundObservation
        Next observation after replay.
    """

    return replay_observation(agent, rounds)


def replay_observation(agent: AgentProtocol, rounds: list[dict]) -> RoundObservation:
    """Rebuild agent internal state from persisted round history.

    This function resets the agent and replays all prior transitions, producing
    the next observation to use for action selection.

    Parameters
    ----------
    agent : AgentProtocol
        Agent to reset and replay.
    rounds : list[dict]
        Historical rows with ``player_action``, ``ai_action``, ``reward_delta``
        keys from the player's perspective.

    Returns
    -------
    RoundObservation
        Observation representing the next step after replay.
    """

    agent.reset(seed=None)
    cumulative_reward = 0
    last_opponent_action: int | None = None
    for index, row in enumerate(rounds):
        obs = RoundObservation(
            step=index,
            last_opponent_action=last_opponent_action,
            cumulative_reward=cumulative_reward,
        )
        player_reward = int(row["reward_delta"])
        ai_reward = -player_reward
        transition = RoundTransition(
            observation=obs,
            action=int(row["ai_action"]),
            opponent_action=int(row["player_action"]),
            outcome=reward_to_outcome(player_reward),
            reward_delta=ai_reward,
            round_index=index,
        )
        agent.observe(transition)
        cumulative_reward += ai_reward
        last_opponent_action = int(row["player_action"])
    return RoundObservation(
        step=len(rounds),
        last_opponent_action=last_opponent_action,
        cumulative_reward=cumulative_reward,
    )


def play_human_round(agent: AgentProtocol, player_action: int, rounds: list[dict]) -> RoundResult:
    """Play one human-vs-agent round from persisted history.

    Parameters
    ----------
    agent : AgentProtocol
        Agent implementation.
    player_action : int
        Human-selected action.
    rounds : list[dict]
        Prior round history for this session.

    Returns
    -------
    RoundResult
        Result payload from the player's perspective.
    """

    observation = replay_observation(agent, rounds)
    result, _ = play_human_round_stateful(agent, player_action=player_action, observation=observation)
    return result


def play_human_round_stateful(
    agent: AgentProtocol,
    player_action: int,
    observation: RoundObservation,
) -> tuple[RoundResult, RoundObservation]:
    """Play one round using an already-hydrated runtime observation.

    This is the low-latency path used when runtime state is cached and replay
    is unnecessary.

    Parameters
    ----------
    agent : AgentProtocol
        Agent ready to act on the provided observation.
    player_action : int
        Human-selected action.
    observation : RoundObservation
        Current agent observation before action selection.

    Returns
    -------
    tuple[RoundResult, RoundObservation]
        A pair of:

        - ``RoundResult`` for API/UI response
        - next ``RoundObservation`` for subsequent step
    """

    player = int(normalize_action(player_action))
    ai_action = int(normalize_action(agent.select_action(observation)))
    player_reward = score_round(player, ai_action)
    outcome = reward_to_outcome(player_reward)
    transition = RoundTransition(
        observation=observation,
        action=ai_action,
        opponent_action=player,
        outcome=outcome,
        reward_delta=-player_reward,
        round_index=observation.step,
    )
    agent.observe(transition)
    result = RoundResult(
        player_action=player,
        opponent_action=ai_action,
        outcome=outcome,
        reward_delta=player_reward,
        round_index=observation.step,
    )
    next_observation = RoundObservation(
        step=observation.step + 1,
        last_opponent_action=player,
        cumulative_reward=observation.cumulative_reward - player_reward,
    )
    return result, next_observation
