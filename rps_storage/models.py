from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class GameRecord:
    id: int
    agent_name: str
    session_index: int
    rounds_played: int
    score_player: int
    score_ai: int
    score_ties: int
    created_at: str
    updated_at: str


@dataclass(slots=True)
class TrainingJobRecord:
    id: int
    status: str
    model_type: str
    params_json: str
    progress: float
    metrics_json: str | None
    error_message: str | None
    model_id: int | None
    created_at: str
    updated_at: str


@dataclass(slots=True)
class ModelRecord:
    id: int
    name: str
    model_type: str
    artifact_path: str
    lookback: int
    metrics_json: str | None
    is_active: int
    created_at: str


@dataclass(slots=True)
class RLJobRecord:
    id: int
    status: str
    params_json: str
    progress: float
    metrics_json: str | None
    error_message: str | None
    model_id: int | None
    created_at: str
    updated_at: str
