from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from rps_web import create_app


@pytest.fixture
def app(tmp_path: Path):
    app = create_app(
        {
            "TESTING": True,
            "DB_PATH": str(tmp_path / "test.db"),
            "EVENTS_DIR": str(tmp_path / "events"),
            "MODELS_DIR": str(tmp_path / "models"),
            "EXPORTS_DIR": str(tmp_path / "exports"),
            "DEFAULT_AGENT": "markov",
        }
    )
    yield app


@pytest.fixture
def client(app):
    return app.test_client()


def _create_game(client, agent: str) -> int:
    response = client.post("/api/v1/games", json={"agent": agent})
    assert response.status_code == 201
    return int(response.get_json()["game"]["game_id"])


def test_create_round_and_reset_preserves_round_history(app, client):
    game_id = _create_game(client, "copy_opponent")
    first = client.post(f"/api/v1/games/{game_id}/round", json={"action": "rock"})
    assert first.status_code == 200
    first_round = first.get_json()["round"]
    assert int(first_round["server_elapsed_ms"]) >= 0
    timings = first_round["timings_ms"]
    assert set(timings.keys()) == {"cache_or_load", "agent_step", "persist", "event_log"}
    second = client.post(f"/api/v1/games/{game_id}/round", json={"action": "paper"})
    assert second.status_code == 200

    before_reset = client.get(f"/api/v1/games/{game_id}")
    assert before_reset.status_code == 200
    assert before_reset.get_json()["game"]["rounds_played"] == 2

    reset = client.post(f"/api/v1/games/{game_id}/reset")
    assert reset.status_code == 200
    body = reset.get_json()
    assert body["game"]["rounds_played"] == 0
    assert body["game"]["score_player"] == 0
    assert body["game"]["score_ai"] == 0

    rounds = app.extensions["repository"].list_rounds(game_id)
    assert len(rounds) == 2


def test_agent_vs_agent_match_endpoint_returns_trace(client):
    response = client.post(
        "/api/v1/matches",
        json={
            "agent_a": "rock",
            "agent_b": "scissors",
            "rounds": 6,
            "seed": 11,
        },
    )
    assert response.status_code == 200
    match = response.get_json()["match"]
    assert match["mode"] == "agent_vs_agent"
    assert match["winner"] == "agent_a"
    assert match["score_agent_a"] == 6
    assert match["score_agent_b"] == 0
    assert match["score_ties"] == 0
    assert len(match["trace"]) == 6
    first_round = match["trace"][0]
    assert first_round["agent_a_action_name"] == "rock"
    assert first_round["agent_b_action_name"] == "scissors"
    assert first_round["winner"] == "agent_a"


def test_arena_match_job_lifecycle_persists_trace(client):
    create = client.post(
        "/api/v1/arena/matches",
        json={
            "agent_a": "rock",
            "agent_b": "scissors",
            "rounds": 5,
            "seed": 17,
        },
    )
    assert create.status_code == 202
    match_id = int(create.get_json()["match"]["id"])

    status = None
    trace = None
    for _ in range(60):
        poll = client.get(f"/api/v1/arena/matches/{match_id}")
        assert poll.status_code == 200
        match = poll.get_json()["match"]
        status = match["status"]
        trace = match["trace"]
        if status in {"completed", "failed"}:
            break
        time.sleep(0.05)

    assert status == "completed"
    assert trace is not None
    assert len(trace) == 5
    assert trace[0]["winner"] == "agent_a"


def test_training_job_lifecycle_and_model_activation(app, client):
    game_id = _create_game(client, "rock")
    for step in range(16):
        action = ("rock", "paper", "scissors")[step % 3]
        response = client.post(f"/api/v1/games/{game_id}/round", json={"action": action})
        assert response.status_code == 200

    create_job = client.post(
        "/api/v1/training/jobs",
        json={
            "model_type": "frequency",
            "lookback": 3,
            "test_size": 0.25,
            "learning_rate": 0.001,
            "hidden_layer_sizes": [32, 16],
            "max_iter": 100,
            "random_state": 12,
        },
    )
    assert create_job.status_code == 202
    job_id = int(create_job.get_json()["job"]["id"])

    status = None
    for _ in range(60):
        poll = client.get(f"/api/v1/training/jobs/{job_id}")
        assert poll.status_code == 200
        status = poll.get_json()["job"]["status"]
        if status in {"completed", "failed"}:
            break
        time.sleep(0.1)
    assert status == "completed"

    models_response = client.get("/api/v1/models")
    assert models_response.status_code == 200
    models = models_response.get_json()["models"]
    assert models
    model_id = int(models[0]["id"])

    activate = client.post(f"/api/v1/models/{model_id}/activate")
    assert activate.status_code == 200
    assert activate.get_json()["model"]["is_active"] is True


def test_internal_training_run_endpoint_enforces_token(app, client):
    game_id = _create_game(client, "rock")
    for step in range(12):
        action = ("rock", "paper", "scissors")[step % 3]
        response = client.post(f"/api/v1/games/{game_id}/round", json={"action": action})
        assert response.status_code == 200

    create_job = client.post("/api/v1/training/jobs", json={"model_type": "frequency", "lookback": 3})
    assert create_job.status_code == 202
    job_id = int(create_job.get_json()["job"]["id"])

    app.config["INTERNAL_WORKER_TOKEN"] = "test-token"
    denied = client.post(f"/api/v1/internal/training/jobs/{job_id}/run")
    assert denied.status_code == 403

    allowed = client.post(
        f"/api/v1/internal/training/jobs/{job_id}/run",
        headers={"X-Worker-Token": "test-token"},
    )
    assert allowed.status_code == 200
    assert int(allowed.get_json()["job"]["id"]) == job_id


def test_benchmark_endpoint_returns_canonical_results(client):
    response = client.post("/api/v1/benchmarks/run", json={"agent": "markov", "rounds": 120, "seed": 9})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["agent"] == "markov"
    benchmark = payload["benchmark"]
    assert benchmark["suite"] == "core"
    assert "overall_non_tie_win_rate" in benchmark
    bots = {item["bot"] for item in benchmark["results"]}
    assert bots == {"quincy", "abbey", "kris", "mrugesh"}


def test_benchmark_endpoint_supports_extended_suite(client):
    response = client.post(
        "/api/v1/benchmarks/run",
        json={"agent": "markov", "rounds": 100, "seed": 9, "suite": "extended"},
    )
    assert response.status_code == 200
    benchmark = response.get_json()["benchmark"]
    assert benchmark["suite"] == "extended"
    bots = {item["bot"] for item in benchmark["results"]}
    assert {
        "quincy",
        "abbey",
        "kris",
        "mrugesh",
        "random",
        "rock",
        "paper",
        "scissors",
        "nash_equilibrium",
        "switcher",
    } <= bots


def test_benchmark_suite_listing_endpoint(client):
    response = client.get("/api/v1/benchmarks/suites")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["default_suite"] == "core"
    assert "core" in payload["suites"]
    assert "extended" in payload["suites"]


def test_benchmark_endpoint_rejects_unknown_suite(client):
    response = client.post("/api/v1/benchmarks/run", json={"agent": "markov", "rounds": 120, "suite": "bogus"})
    assert response.status_code == 400
    payload = response.get_json()
    assert payload is not None
    assert "Unknown benchmark suite" in payload["error"]


def test_benchmark_endpoint_returns_timeout_for_heavy_workload(client, monkeypatch):
    def _slow(*args, **kwargs):
        raise TimeoutError("Benchmark exceeded time budget.")

    monkeypatch.setattr("rps_web.blueprints.benchmarks.benchmark_agent", _slow)
    response = client.post("/api/v1/benchmarks/run", json={"agent": "markov", "rounds": 120, "suite": "core"})
    assert response.status_code == 408
    payload = response.get_json()
    assert payload is not None
    assert "time budget" in payload["error"].lower()


def test_benchmark_endpoint_returns_json_error_on_internal_failure(client, monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("rps_web.blueprints.benchmarks.benchmark_agent", _boom)
    response = client.post("/api/v1/benchmarks/run", json={"agent": "markov", "rounds": 120, "seed": 9})
    assert response.status_code == 500
    payload = response.get_json()
    assert payload is not None
    assert "error" in payload
    assert "Benchmark failed:" in payload["error"]


def test_training_readiness_endpoint(client):
    response = client.get("/api/v1/training/readiness?lookback=5")
    assert response.status_code == 200
    readiness = response.get_json()["readiness"]
    assert "sample_count" in readiness
    assert "can_train" in readiness
    assert "sklearn_available" in readiness


def test_latency_telemetry_endpoint_writes_event(app, client):
    response = client.post(
        "/api/v1/telemetry/latency",
        json={
            "game_id": 1,
            "round_id": 2,
            "round_index": 1,
            "agent_name": "markov",
            "client_elapsed_ms": 432,
            "server_elapsed_ms": 321,
            "timings_ms": {"cache_or_load": 5, "agent_step": 8, "persist": 12},
        },
    )
    assert response.status_code == 204

    latency_dir = Path(app.config["EVENTS_DIR"]) / "latency"
    files = sorted(latency_dir.glob("*.jsonl"))
    assert files
    lines = [line for line in files[-1].read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines
    payload = json.loads(lines[-1])
    assert payload["event_type"] == "round_latency"
    assert payload["client_elapsed_ms"] == 432
    assert payload["server_elapsed_ms"] == 321


def test_worker_token_falls_back_to_internal_token(tmp_path: Path):
    app = create_app(
        {
            "TESTING": True,
            "DB_PATH": str(tmp_path / "fallback.db"),
            "EVENTS_DIR": str(tmp_path / "events"),
            "MODELS_DIR": str(tmp_path / "models"),
            "EXPORTS_DIR": str(tmp_path / "exports"),
            "TRAINING_EXECUTION_MODE": "task_queue",
            "TRAINING_WORKER_TOKEN": "",
            "INTERNAL_WORKER_TOKEN": "abc123",
            "TASKS_PROJECT_ID": "p",
            "TASKS_LOCATION": "us-west3",
            "TASKS_QUEUE": "q",
            "TRAINING_WORKER_URL": "https://example.com",
        }
    )
    manager = app.extensions["training_jobs"]
    assert manager.worker_token == "abc123"


def test_create_app_loads_database_url_from_secret(monkeypatch, tmp_path: Path):
    def _fake_read_secret(version_name: str) -> str:
        assert version_name == "projects/p/secrets/db-url/versions/latest"
        return f"sqlite+pysqlite:///{(tmp_path / 'secret.db').as_posix()}"

    monkeypatch.setattr("rps_web._read_secret_version", _fake_read_secret)
    app = create_app(
        {
            "TESTING": True,
            "DATABASE_URL": "",
            "DATABASE_URL_SECRET": "projects/p/secrets/db-url/versions/latest",
            "EVENTS_DIR": str(tmp_path / "events"),
            "MODELS_DIR": str(tmp_path / "models"),
            "EXPORTS_DIR": str(tmp_path / "exports"),
        }
    )
    assert str(app.config["DATABASE_URL"]).startswith("sqlite+pysqlite:///")


def test_rl_job_lifecycle_creates_model(client):
    create_job = client.post(
        "/api/v1/rl/jobs",
        json={
            "episodes": 20,
            "steps_per_episode": 40,
            "alpha": 0.2,
            "gamma": 0.9,
            "epsilon_start": 0.8,
            "epsilon_end": 0.1,
            "epsilon_decay": 0.99,
            "seed": 5,
            "opponents": ["rock", "paper", "scissors"],
        },
    )
    assert create_job.status_code == 202
    job_id = int(create_job.get_json()["job"]["id"])

    status = None
    model_id = None
    for _ in range(80):
        poll = client.get(f"/api/v1/rl/jobs/{job_id}")
        assert poll.status_code == 200
        body = poll.get_json()["job"]
        status = body["status"]
        model_id = body["model_id"]
        if status in {"completed", "failed"}:
            break
        time.sleep(0.1)
    assert status == "completed"
    assert model_id is not None

    activate = client.post(f"/api/v1/models/{model_id}/activate")
    assert activate.status_code == 200
    play_game = client.post("/api/v1/games", json={"agent": "active_model"})
    assert play_game.status_code == 201
