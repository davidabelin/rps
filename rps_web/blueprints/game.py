"""Gameplay API routes for human-vs-agent interactions."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from flask import Blueprint, current_app, jsonify, request

from rps_agents import ModelBackedAgent, build_heuristic_agent, list_agent_specs
from rps_core.engine import play_human_round_stateful, replay_observation
from rps_core.matches import play_agent_match
from rps_core.scoring import ACTION_NAMES
from rps_core.types import normalize_action
from rps_storage.object_store import is_gcs_uri, join_storage_path
from rps_training.dataset import append_round_event
from rps_web.runtime import GameRuntimeState

game_bp = Blueprint("game_api", __name__, url_prefix="/api/v1")


def _repo():
    """Return storage repository extension."""

    return current_app.extensions["repository"]


def _runtime():
    """Return in-memory game runtime cache extension."""

    return current_app.extensions["game_runtime"]


def _serialize_game(game: dict) -> dict:
    """Serialize database game row into API response shape."""

    return {
        "game_id": int(game["id"]),
        "agent_name": game["agent_name"],
        "session_index": int(game["session_index"]),
        "rounds_played": int(game["rounds_played"]),
        "score_player": int(game["score_player"]),
        "score_ai": int(game["score_ai"]),
        "score_ties": int(game["score_ties"]),
        "updated_at": game["updated_at"],
    }


def _available_agent_names() -> list[str]:
    return [spec.name for spec in list_agent_specs()]


def _default_match_opponent(agent_name: str) -> str:
    for candidate in _available_agent_names():
        if candidate != agent_name:
            return candidate
    return agent_name


def _build_agent_from_name(agent_name: str):
    if agent_name == "active_model":
        model_record = _repo().get_active_model()
        if model_record is None:
            raise RuntimeError("No active model is available. Activate a trained model first.")
        return ModelBackedAgent(str(model_record["artifact_path"]))
    available = set(_available_agent_names())
    if agent_name not in available:
        raise KeyError(f"Unknown agent '{agent_name}'.")
    return build_heuristic_agent(agent_name)


def _resolve_agent_factory_and_signature(game: dict):
    """Resolve agent factory and cache signature for a game row.

    Parameters
    ----------
    game : dict
        Game row from repository.

    Returns
    -------
    tuple[callable, str]
        Factory callable and stable signature string used for runtime cache
        invalidation when active model changes.
    """

    if game["agent_name"] == "active_model":
        model_record = _repo().get_active_model()
        if model_record is None:
            raise RuntimeError("No active model is available. Activate a trained model first.")
        model_id = int(model_record["id"])
        artifact_path = str(model_record["artifact_path"])
        return (lambda: ModelBackedAgent(artifact_path), f"active_model:{model_id}")
    name = str(game["agent_name"])
    return (lambda: build_heuristic_agent(name), f"heuristic:{name}")


def _should_write_round_event() -> bool:
    """Decide whether per-round event log files should be written.

    Returns
    -------
    bool
        ``True`` when round event logging is enabled by configuration and the
        selected storage mode is appropriate for per-round writes.
    """

    mode = str(current_app.config.get("ROUND_EVENT_LOGGING_MODE", "auto")).strip().lower()
    if mode in {"off", "disabled", "false", "0", "none", "db_only"}:
        return False
    if mode in {"on", "enabled", "true", "1", "always"}:
        return True
    events_dir = str(current_app.config.get("EVENTS_DIR", ""))
    return bool(events_dir) and not is_gcs_uri(events_dir)


def _should_write_latency_event() -> bool:
    """Decide whether latency telemetry should be persisted."""

    mode = str(current_app.config.get("LATENCY_EVENT_LOGGING_MODE", "on")).strip().lower()
    return mode not in {"off", "disabled", "false", "0", "none"}


def _latency_events_dir(events_dir: str) -> str:
    """Build latency-event destination path from configured ``EVENTS_DIR``."""

    if is_gcs_uri(events_dir):
        return join_storage_path(events_dir, "latency")
    return str(Path(events_dir) / "latency")


def _load_runtime_state(game: dict) -> GameRuntimeState:
    """Load or build a low-latency runtime state for a game session.

    Parameters
    ----------
    game : dict
        Game row from repository.

    Returns
    -------
    GameRuntimeState
        Cached or newly hydrated runtime state.
    """

    game_id = int(game["id"])
    session_index = int(game["session_index"])
    cached = _runtime().get(game_id)
    if cached is not None and cached.session_index == session_index:
        # Keep a stable per-session policy instance for active_model games.
        # This avoids a model-registry lookup on every round request.
        if str(game["agent_name"]) == "active_model":
            if cached.signature.startswith("active_model:"):
                return cached
        else:
            return cached
    agent_factory, signature = _resolve_agent_factory_and_signature(game)
    agent = agent_factory()
    history = _repo().list_rounds(game_id, session_index=session_index)
    observation = replay_observation(agent, history)
    state = GameRuntimeState(
        game_id=game_id,
        agent_name=str(game["agent_name"]),
        session_index=session_index,
        signature=signature,
        agent=agent,
        observation=observation,
    )
    _runtime().put(state)
    return state


@game_bp.get("/agents")
def list_agents():
    """List playable agents available to UI/API clients."""

    specs = list_agent_specs()
    models = _repo().list_models(limit=1)
    payload = [
        {
            "name": spec.name,
            "description": spec.description,
            "type": "heuristic",
        }
        for spec in specs
    ]
    if models:
        payload.append(
            {
                "name": "active_model",
                "description": "Use the model marked active in the model registry.",
                "type": "trained",
            }
        )
    return jsonify({"agents": payload})


@game_bp.post("/games")
def create_game():
    """Create a new game session with selected agent/model."""

    payload = request.get_json(silent=True) or {}
    agent_name = str(payload.get("agent", current_app.config["DEFAULT_AGENT"]))
    available = set(_available_agent_names())
    if agent_name != "active_model" and agent_name not in available:
        return jsonify({"error": f"Unknown agent '{agent_name}'."}), 400
    if agent_name == "active_model" and _repo().get_active_model() is None:
        return jsonify({"error": "No active model is available. Train and activate a model first."}), 400
    game = _repo().create_game(agent_name=agent_name)
    _runtime().forget_game(int(game["id"]))
    return jsonify({"game": _serialize_game(game)}), 201


@game_bp.get("/games/<int:game_id>")
def get_game(game_id: int):
    """Return current score/state for one game id."""

    game = _repo().get_game(game_id)
    if game is None:
        return jsonify({"error": "Game not found."}), 404
    return jsonify({"game": _serialize_game(game)})


@game_bp.post("/games/<int:game_id>/round")
def play_round(game_id: int):
    """Play one round for the selected game and persist the outcome.

    Parameters
    ----------
    game_id : int
        Target game identifier.

    Returns
    -------
    flask.Response
        JSON response with updated game score and round result payload.
    """

    started = perf_counter()
    stage_started = started
    payload = request.get_json(silent=True) or {}
    if "action" not in payload:
        return jsonify({"error": "Request body must include 'action'."}), 400
    try:
        player_action = int(normalize_action(payload["action"]))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    runtime_state = _runtime().get(game_id)
    game = None
    if runtime_state is None:
        game = _repo().get_game(game_id)
        if game is None:
            return jsonify({"error": "Game not found."}), 404
        try:
            runtime_state = _load_runtime_state(game)
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 400
    cache_and_load_ms = int(round((perf_counter() - stage_started) * 1000))

    stage_started = perf_counter()
    result, next_observation = play_human_round_stateful(
        agent=runtime_state.agent,
        player_action=player_action,
        observation=runtime_state.observation,
    )
    runtime_state.observation = next_observation
    _runtime().put(runtime_state)
    agent_step_ms = int(round((perf_counter() - stage_started) * 1000))

    stage_started = perf_counter()
    try:
        stored_round, updated_game = _repo().record_round_and_update_game(
            game_id=game_id,
            session_index=int(runtime_state.session_index),
            round_index=result.round_index,
            player_action=result.player_action,
            ai_action=result.opponent_action,
            outcome=result.outcome,
            reward_delta=result.reward_delta,
        )
    except KeyError:
        _runtime().forget_game(game_id)
        return jsonify({"error": "Game session changed. Start a new game and try again."}), 409
    persist_ms = int(round((perf_counter() - stage_started) * 1000))

    stage_started = perf_counter()
    if _should_write_round_event():
        append_round_event(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "game_id": int(game_id),
                "session_index": int(runtime_state.session_index),
                "round_index": int(result.round_index),
                "agent_name": runtime_state.agent_name,
                "player_action": int(result.player_action),
                "ai_action": int(result.opponent_action),
                "player_action_name": ACTION_NAMES[int(result.player_action)],
                "ai_action_name": ACTION_NAMES[int(result.opponent_action)],
                "outcome": result.outcome,
                "reward_delta": int(result.reward_delta),
            },
            events_dir=current_app.config["EVENTS_DIR"],
        )
    event_log_ms = int(round((perf_counter() - stage_started) * 1000))

    return jsonify(
        {
            "game": _serialize_game(updated_game),
            "round": {
                "id": int(stored_round["id"]),
                "round_index": int(result.round_index),
                "player_action": int(result.player_action),
                "ai_action": int(result.opponent_action),
                "player_action_name": ACTION_NAMES[int(result.player_action)],
                "ai_action_name": ACTION_NAMES[int(result.opponent_action)],
                "outcome": result.outcome,
                "reward_delta": int(result.reward_delta),
                "server_elapsed_ms": int(round((perf_counter() - started) * 1000)),
                "timings_ms": {
                    "cache_or_load": cache_and_load_ms,
                    "agent_step": agent_step_ms,
                    "persist": persist_ms,
                    "event_log": event_log_ms,
                },
            },
        }
    )


@game_bp.post("/games/<int:game_id>/reset")
def reset_game(game_id: int):
    """Reset a game score while preserving historical rounds."""

    updated_game = _repo().reset_game(game_id)
    if updated_game is None:
        return jsonify({"error": "Game not found."}), 404
    _runtime().forget_game(game_id)
    return jsonify({"game": _serialize_game(updated_game)})


@game_bp.post("/matches")
def run_match():
    """Run one non-persisted agent-vs-agent match and return a replay trace."""

    payload = request.get_json(silent=True) or {}
    agent_a_name = str(payload.get("agent_a", current_app.config["DEFAULT_AGENT"]))
    agent_b_name = str(payload.get("agent_b", _default_match_opponent(agent_a_name)))
    try:
        rounds = int(payload.get("rounds", current_app.config.get("AGENT_MATCH_DEFAULT_ROUNDS", 50)))
    except (TypeError, ValueError):
        return jsonify({"error": "rounds must be an integer."}), 400
    if rounds <= 0 or rounds > 5000:
        return jsonify({"error": "rounds must be between 1 and 5000."}), 400

    raw_seed = payload.get("seed")
    try:
        seed = int(raw_seed) if raw_seed is not None else None
    except (TypeError, ValueError):
        return jsonify({"error": "seed must be an integer when provided."}), 400

    try:
        agent_a = _build_agent_from_name(agent_a_name)
        agent_b = _build_agent_from_name(agent_b_name)
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 400

    match = play_agent_match(
        agent_a=agent_a,
        agent_b=agent_b,
        agent_a_name=agent_a_name,
        agent_b_name=agent_b_name,
        rounds=rounds,
        seed=seed,
    )
    return jsonify({"match": match})


@game_bp.post("/telemetry/latency")
def record_latency_telemetry():
    """Persist one client-side latency sample as JSONL telemetry."""

    payload = request.get_json(silent=True) or {}
    try:
        game_id = int(payload["game_id"])
        round_id = int(payload["round_id"])
        round_index = int(payload["round_index"])
        client_elapsed_ms = int(payload["client_elapsed_ms"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Payload must include game_id, round_id, round_index, client_elapsed_ms."}), 400
    server_elapsed_ms = payload.get("server_elapsed_ms")
    timings_ms = payload.get("timings_ms")
    agent_name = str(payload.get("agent_name", ""))
    if not _should_write_latency_event():
        return ("", 204)
    events_dir = str(current_app.config.get("EVENTS_DIR", "")).strip()
    if not events_dir:
        return ("", 204)
    event = {
        "timestamp": datetime.now(UTC).isoformat(),
        "event_type": "round_latency",
        "game_id": game_id,
        "round_id": round_id,
        "round_index": round_index,
        "agent_name": agent_name,
        "client_elapsed_ms": client_elapsed_ms,
        "server_elapsed_ms": int(server_elapsed_ms) if isinstance(server_elapsed_ms, (int, float)) else None,
        "timings_ms": timings_ms if isinstance(timings_ms, dict) else None,
        "user_agent": request.headers.get("User-Agent", ""),
        "source": "web_play",
    }
    append_round_event(event, events_dir=_latency_events_dir(events_dir))
    return ("", 204)
