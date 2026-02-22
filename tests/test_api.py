from __future__ import annotations

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
    assert "overall_non_tie_win_rate" in benchmark
    bots = {item["bot"] for item in benchmark["results"]}
    assert bots == {"quincy", "abbey", "kris", "mrugesh"}


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
