"""Unified CLI for RPS gameplay, data collection, and model training experiments."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from random import Random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rps_agents import ModelBackedAgent, build_heuristic_agent, list_agent_specs
from rps_core.engine import play_human_round_stateful, replay_observation
from rps_core.scoring import ACTION_NAMES
from rps_core.types import normalize_action
from rps_rl.trainer import RLTrainConfig, train_q_policy
from rps_storage import RPSRepository
from rps_storage.object_store import is_gcs_uri, join_storage_path
from rps_training.supervised import TrainConfig, train_model


def _repository(db_path: str) -> RPSRepository:
    repo = RPSRepository(db_path)
    repo.init_schema()
    return repo


def _resolve_agent(repo: RPSRepository, agent_name: str):
    if agent_name == "active_model":
        model = repo.get_active_model()
        if model is None:
            raise RuntimeError("No active model is available. Activate a model first.")
        return ModelBackedAgent(str(model["artifact_path"]))
    return build_heuristic_agent(agent_name)


def _parse_hidden_layers(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in str(raw).split(",") if part.strip())
    return values or (64, 32)


def _resolve_batch_size(raw: str):
    token = str(raw).strip().lower()
    if token in {"", "auto"}:
        return "auto"
    return int(token)


def _player_action_for_step(args, step_index: int, rng: Random):
    if args.interactive:
        while True:
            raw = input(f"Round {step_index + 1} action [rock/paper/scissors or r/p/s, q=quit]: ").strip().lower()
            if raw in {"q", "quit", "exit"}:
                return None
            try:
                return int(normalize_action(raw))
            except ValueError:
                print("Invalid action.")
    if args.actions:
        sequence = [part.strip() for part in str(args.actions).split(",") if part.strip()]
        if not sequence:
            return int(rng.randint(0, 2))
        return int(normalize_action(sequence[step_index % len(sequence)]))
    return int(rng.randint(0, 2))


def cmd_agents(args) -> int:
    print("Registered heuristic agents:")
    for spec in list_agent_specs():
        print(f"- {spec.name}: {spec.description}")
    print("- active_model: currently activated trained model")
    return 0


def cmd_models(args) -> int:
    repo = _repository(args.db_path)
    rows = repo.list_models(limit=args.limit)
    if not rows:
        print("No models found.")
        return 0
    print("Models:")
    for row in rows:
        active = "active" if bool(row["is_active"]) else "inactive"
        print(
            f"- id={row['id']} name={row['name']} type={row['model_type']} "
            f"lookback={row['lookback']} status={active}"
        )
    return 0


def cmd_play(args) -> int:
    repo = _repository(args.db_path)
    available = {spec.name for spec in list_agent_specs()}
    if args.agent != "active_model" and args.agent not in available:
        raise ValueError(f"Unknown agent '{args.agent}'. Run `agents` to list available names.")

    agent = _resolve_agent(repo, args.agent)
    game = repo.create_game(agent_name=args.agent)
    game_id = int(game["id"])
    session_index = int(game["session_index"])
    history = repo.list_rounds(game_id, session_index=session_index)
    observation = replay_observation(agent, history)
    rng = Random(args.seed)

    print(f"Game {game_id} started vs {args.agent}.")
    for step in range(args.rounds):
        player_action = _player_action_for_step(args, step, rng)
        if player_action is None:
            print("Stopped by user.")
            break
        result, next_observation = play_human_round_stateful(
            agent=agent,
            player_action=player_action,
            observation=observation,
        )
        stored_round, updated_game = repo.record_round_and_update_game(
            game_id=game_id,
            session_index=session_index,
            round_index=result.round_index,
            player_action=result.player_action,
            ai_action=result.opponent_action,
            outcome=result.outcome,
            reward_delta=result.reward_delta,
        )
        observation = next_observation
        print(
            f"#{int(stored_round['round_index']) + 1} you={ACTION_NAMES[int(result.player_action)]} "
            f"ai={ACTION_NAMES[int(result.opponent_action)]} outcome={result.outcome} "
            f"score={updated_game['score_player']}-{updated_game['score_ai']}-{updated_game['score_ties']}"
        )

    final_game = repo.get_game(game_id) or game
    print(
        f"Final score game_id={game_id}: "
        f"{final_game['score_player']}-{final_game['score_ai']}-{final_game['score_ties']} "
        f"over {final_game['rounds_played']} rounds."
    )
    return 0


def cmd_train_supervised(args) -> int:
    repo = _repository(args.db_path)
    rounds = repo.list_rounds_for_training()
    config = TrainConfig(
        model_type=args.model_type,
        lookback=args.lookback,
        test_size=args.test_size,
        learning_rate=args.learning_rate,
        hidden_layer_sizes=_parse_hidden_layers(args.hidden_layers),
        epochs=args.epochs,
        batch_size=_resolve_batch_size(args.batch_size),
        random_state=args.random_state,
    )
    models_dir = str(args.models_dir)
    if not is_gcs_uri(models_dir):
        Path(models_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    artifact_path = join_storage_path(models_dir, f"{config.model_type}_cli_{timestamp}.pkl")
    metrics = train_model(rounds=rounds, config=config, artifact_path=artifact_path)
    model_name = args.name or f"{config.model_type}-cli-{timestamp}"
    model = repo.create_model(
        name=model_name,
        model_type=config.model_type,
        artifact_path=artifact_path,
        lookback=config.lookback,
        metrics=metrics,
    )
    if args.activate:
        repo.activate_model(int(model["id"]))
    print(f"Supervised training complete. model_id={model['id']} artifact={artifact_path}")
    print(f"Test accuracy={metrics.get('test_accuracy', 0.0):.4f}")
    return 0


def cmd_train_rl(args) -> int:
    repo = _repository(args.db_path)
    opponents = tuple(part.strip() for part in str(args.opponents).split(",") if part.strip())
    config = RLTrainConfig(
        episodes=args.episodes,
        steps_per_episode=args.steps_per_episode,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
        opponents=opponents or RLTrainConfig().opponents,
        opponent_schedule=args.opponent_schedule,
    )
    models_dir = str(args.models_dir)
    if not is_gcs_uri(models_dir):
        Path(models_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    artifact_path = join_storage_path(models_dir, f"rl_qtable_cli_{timestamp}.pkl")
    metrics = train_q_policy(config=config, artifact_path=artifact_path)
    model_name = args.name or f"rl-qtable-cli-{timestamp}"
    model = repo.create_model(
        name=model_name,
        model_type="rl_qtable",
        artifact_path=artifact_path,
        lookback=1,
        metrics=metrics,
    )
    if args.activate:
        repo.activate_model(int(model["id"]))
    print(f"RL training complete. model_id={model['id']} artifact={artifact_path}")
    print(f"Final 50 non-tie win rate={metrics.get('final_50_non_tie_win_rate', 0.0):.4f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RPS CLI for play/data collection/training experiments.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    agents_p = sub.add_parser("agents", help="List available agent names.")
    agents_p.set_defaults(fn=cmd_agents)

    models_p = sub.add_parser("models", help="List model registry rows.")
    models_p.add_argument("--db-path", default="data/rps.db")
    models_p.add_argument("--limit", type=int, default=30)
    models_p.set_defaults(fn=cmd_models)

    play_p = sub.add_parser("play", help="Play text-only rounds and persist them.")
    play_p.add_argument("--db-path", default="data/rps.db")
    play_p.add_argument("--agent", default="markov")
    play_p.add_argument("--rounds", type=int, default=30)
    play_p.add_argument("--seed", type=int, default=7)
    play_p.add_argument("--interactive", action="store_true")
    play_p.add_argument(
        "--actions",
        default="",
        help="Comma-separated scripted actions (example: rock,paper,scissors). If unset, random actions are used unless --interactive.",
    )
    play_p.set_defaults(fn=cmd_play)

    sup_p = sub.add_parser("train-supervised", help="Run supervised training with explicit hyperparameters.")
    sup_p.add_argument("--db-path", default="data/rps.db")
    sup_p.add_argument("--models-dir", default="data/models")
    sup_p.add_argument("--name", default="")
    sup_p.add_argument("--activate", action="store_true")
    sup_p.add_argument("--model-type", choices=("decision_tree", "mlp", "frequency"), default="decision_tree")
    sup_p.add_argument("--lookback", type=int, default=5)
    sup_p.add_argument("--test-size", type=float, default=0.2)
    sup_p.add_argument("--learning-rate", type=float, default=0.001)
    sup_p.add_argument("--hidden-layers", default="64,32")
    sup_p.add_argument("--epochs", type=int, default=200)
    sup_p.add_argument("--batch-size", default="auto")
    sup_p.add_argument("--random-state", type=int, default=42)
    sup_p.set_defaults(fn=cmd_train_supervised)

    rl_p = sub.add_parser("train-rl", help="Run tabular RL training with configurable opponent pool/schedule.")
    rl_p.add_argument("--db-path", default="data/rps.db")
    rl_p.add_argument("--models-dir", default="data/models")
    rl_p.add_argument("--name", default="")
    rl_p.add_argument("--activate", action="store_true")
    rl_p.add_argument("--episodes", type=int, default=300)
    rl_p.add_argument("--steps-per-episode", type=int, default=300)
    rl_p.add_argument("--alpha", type=float, default=0.15)
    rl_p.add_argument("--gamma", type=float, default=0.95)
    rl_p.add_argument("--epsilon-start", type=float, default=1.0)
    rl_p.add_argument("--epsilon-end", type=float, default=0.05)
    rl_p.add_argument("--epsilon-decay", type=float, default=0.995)
    rl_p.add_argument("--seed", type=int, default=7)
    rl_p.add_argument(
        "--opponents",
        default="rock,paper,scissors,copy_opponent,reactionary,counter_reactionary,statistical,markov,nash_equilibrium,multi_armed_bandit",
    )
    rl_p.add_argument("--opponent-schedule", choices=("cycle", "curriculum"), default="curriculum")
    rl_p.set_defaults(fn=cmd_train_rl)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
