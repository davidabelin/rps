"""Round execution helpers for human-vs-agent RPS play loops.

Role
----
This module is the pure gameplay layer between web request handling and agent
implementations. It reconstructs agent state from persisted history and
produces round results without knowing anything about Flask, storage, or model
registry concerns.

Cross-Repo Context
------------------
`rps_core.engine` and `c4_core.engine` deliberately fill the same role in their
respective labs: one module owns deterministic gameplay resolution while the
web/API layer owns persistence and runtime caching.
"""

from __future__ import annotations

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

    Notes
    -----
    The name is kept for backward compatibility. New code should think of this
    as observation replay rather than a special hydration protocol.
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

    Used By
    -------
    `play_human_round` and the web layer whenever runtime agent state is not
    already cached.
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

    Role
    ----
    This is the stateless convenience path used when the caller only has stored
    round history and wants a resolved round in one call.
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

    Cross-Repo Context
    ------------------
    This is the RPS equivalent of the low-latency cached-runtime path in
    Connect4. Both labs use it to avoid replaying full history on every move
    when the web layer already holds a warm agent/runtime pair.
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
