from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any


def utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


class RPSRepository:
    def __init__(self, db_path: str) -> None:
        self.db_path = str(db_path)
        self._lock = RLock()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    def init_schema(self) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.executescript(
                    """
                    PRAGMA foreign_keys = ON;

                    CREATE TABLE IF NOT EXISTS games (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        agent_name TEXT NOT NULL,
                        session_index INTEGER NOT NULL DEFAULT 0,
                        rounds_played INTEGER NOT NULL DEFAULT 0,
                        score_player INTEGER NOT NULL DEFAULT 0,
                        score_ai INTEGER NOT NULL DEFAULT 0,
                        score_ties INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS rounds (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        game_id INTEGER NOT NULL,
                        session_index INTEGER NOT NULL,
                        round_index INTEGER NOT NULL,
                        player_action INTEGER NOT NULL,
                        ai_action INTEGER NOT NULL,
                        outcome TEXT NOT NULL,
                        reward_delta INTEGER NOT NULL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY(game_id) REFERENCES games(id)
                    );

                    CREATE TABLE IF NOT EXISTS training_jobs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        status TEXT NOT NULL,
                        model_type TEXT NOT NULL,
                        params_json TEXT NOT NULL,
                        progress REAL NOT NULL DEFAULT 0.0,
                        metrics_json TEXT,
                        error_message TEXT,
                        model_id INTEGER,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS models (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        model_type TEXT NOT NULL,
                        artifact_path TEXT NOT NULL,
                        lookback INTEGER NOT NULL,
                        metrics_json TEXT,
                        is_active INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS model_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_id INTEGER NOT NULL,
                        metric_name TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY(model_id) REFERENCES models(id)
                    );

                    CREATE TABLE IF NOT EXISTS rl_jobs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        status TEXT NOT NULL,
                        params_json TEXT NOT NULL,
                        progress REAL NOT NULL DEFAULT 0.0,
                        metrics_json TEXT,
                        error_message TEXT,
                        model_id INTEGER,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_rounds_game_session ON rounds(game_id, session_index, round_index);
                    CREATE INDEX IF NOT EXISTS idx_models_active ON models(is_active);
                    CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON training_jobs(status);
                    CREATE INDEX IF NOT EXISTS idx_rl_jobs_status ON rl_jobs(status);
                    """
                )

    @staticmethod
    def _row_to_dict(row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        return {key: row[key] for key in row.keys()}

    def create_game(self, agent_name: str) -> dict:
        now = utcnow_iso()
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO games (agent_name, session_index, rounds_played, score_player, score_ai, score_ties, created_at, updated_at)
                    VALUES (?, 0, 0, 0, 0, 0, ?, ?)
                    """,
                    (agent_name, now, now),
                )
                game_id = int(cursor.lastrowid)
                row = conn.execute("SELECT * FROM games WHERE id = ?", (game_id,)).fetchone()
        return self._row_to_dict(row) or {}

    def get_game(self, game_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM games WHERE id = ?", (game_id,)).fetchone()
        return self._row_to_dict(row)

    def update_game_scores(
        self,
        game_id: int,
        rounds_played: int,
        score_player: int,
        score_ai: int,
        score_ties: int,
    ) -> dict:
        now = utcnow_iso()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE games
                    SET rounds_played = ?, score_player = ?, score_ai = ?, score_ties = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (rounds_played, score_player, score_ai, score_ties, now, game_id),
                )
                row = conn.execute("SELECT * FROM games WHERE id = ?", (game_id,)).fetchone()
        return self._row_to_dict(row) or {}

    def reset_game(self, game_id: int) -> dict | None:
        game = self.get_game(game_id)
        if not game:
            return None
        now = utcnow_iso()
        new_session_index = int(game["session_index"]) + 1
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE games
                    SET session_index = ?, rounds_played = 0, score_player = 0, score_ai = 0, score_ties = 0, updated_at = ?
                    WHERE id = ?
                    """,
                    (new_session_index, now, game_id),
                )
                row = conn.execute("SELECT * FROM games WHERE id = ?", (game_id,)).fetchone()
        return self._row_to_dict(row)

    def add_round(
        self,
        game_id: int,
        session_index: int,
        round_index: int,
        player_action: int,
        ai_action: int,
        outcome: str,
        reward_delta: int,
    ) -> dict:
        now = utcnow_iso()
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO rounds
                    (game_id, session_index, round_index, player_action, ai_action, outcome, reward_delta, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (game_id, session_index, round_index, player_action, ai_action, outcome, reward_delta, now),
                )
                row_id = int(cursor.lastrowid)
                row = conn.execute("SELECT * FROM rounds WHERE id = ?", (row_id,)).fetchone()
        return self._row_to_dict(row) or {}

    def list_rounds(self, game_id: int, session_index: int | None = None) -> list[dict]:
        query = "SELECT * FROM rounds WHERE game_id = ?"
        params: list[Any] = [game_id]
        if session_index is not None:
            query += " AND session_index = ?"
            params.append(session_index)
        query += " ORDER BY round_index ASC, id ASC"
        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [self._row_to_dict(row) or {} for row in rows]

    def list_rounds_for_training(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT game_id, session_index, round_index, player_action, ai_action, outcome, reward_delta, created_at
                FROM rounds
                ORDER BY game_id ASC, session_index ASC, round_index ASC
                """
            ).fetchall()
        return [self._row_to_dict(row) or {} for row in rows]

    def create_training_job(self, model_type: str, params: dict) -> dict:
        now = utcnow_iso()
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO training_jobs
                    (status, model_type, params_json, progress, metrics_json, error_message, model_id, created_at, updated_at)
                    VALUES ('queued', ?, ?, 0.0, NULL, NULL, NULL, ?, ?)
                    """,
                    (model_type, json.dumps(params, sort_keys=True), now, now),
                )
                job_id = int(cursor.lastrowid)
                row = conn.execute("SELECT * FROM training_jobs WHERE id = ?", (job_id,)).fetchone()
        return self._row_to_dict(row) or {}

    def update_training_job(
        self,
        job_id: int,
        *,
        status: str | None = None,
        progress: float | None = None,
        metrics: dict | None = None,
        error_message: str | None = None,
        model_id: int | None = None,
    ) -> dict | None:
        updates: list[str] = []
        values: list[Any] = []
        if status is not None:
            updates.append("status = ?")
            values.append(status)
        if progress is not None:
            updates.append("progress = ?")
            values.append(float(progress))
        if metrics is not None:
            updates.append("metrics_json = ?")
            values.append(json.dumps(metrics, sort_keys=True))
        if error_message is not None:
            updates.append("error_message = ?")
            values.append(error_message)
        if model_id is not None:
            updates.append("model_id = ?")
            values.append(int(model_id))
        updates.append("updated_at = ?")
        values.append(utcnow_iso())
        values.append(job_id)
        with self._lock:
            with self._connect() as conn:
                conn.execute(f"UPDATE training_jobs SET {', '.join(updates)} WHERE id = ?", tuple(values))
                row = conn.execute("SELECT * FROM training_jobs WHERE id = ?", (job_id,)).fetchone()
        return self._row_to_dict(row)

    def get_training_job(self, job_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM training_jobs WHERE id = ?", (job_id,)).fetchone()
        return self._row_to_dict(row)

    def list_training_jobs(self, limit: int = 100) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM training_jobs ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_dict(row) or {} for row in rows]

    def create_model(self, name: str, model_type: str, artifact_path: str, lookback: int, metrics: dict) -> dict:
        now = utcnow_iso()
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO models (name, model_type, artifact_path, lookback, metrics_json, is_active, created_at)
                    VALUES (?, ?, ?, ?, ?, 0, ?)
                    """,
                    (name, model_type, artifact_path, lookback, json.dumps(metrics, sort_keys=True), now),
                )
                model_id = int(cursor.lastrowid)
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        conn.execute(
                            """
                            INSERT INTO model_metrics (model_id, metric_name, metric_value, created_at)
                            VALUES (?, ?, ?, ?)
                            """,
                            (model_id, key, float(value), now),
                        )
                row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
        return self._row_to_dict(row) or {}

    def list_models(self, limit: int = 200) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM models ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [self._row_to_dict(row) or {} for row in rows]

    def get_model(self, model_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
        return self._row_to_dict(row)

    def get_active_model(self) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM models WHERE is_active = 1 ORDER BY id DESC LIMIT 1").fetchone()
        return self._row_to_dict(row)

    def activate_model(self, model_id: int) -> dict | None:
        with self._lock:
            with self._connect() as conn:
                conn.execute("UPDATE models SET is_active = 0")
                conn.execute("UPDATE models SET is_active = 1 WHERE id = ?", (model_id,))
                row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
        return self._row_to_dict(row)

    def create_rl_job(self, params: dict) -> dict:
        now = utcnow_iso()
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO rl_jobs
                    (status, params_json, progress, metrics_json, error_message, model_id, created_at, updated_at)
                    VALUES ('queued', ?, 0.0, NULL, NULL, NULL, ?, ?)
                    """,
                    (json.dumps(params, sort_keys=True), now, now),
                )
                job_id = int(cursor.lastrowid)
                row = conn.execute("SELECT * FROM rl_jobs WHERE id = ?", (job_id,)).fetchone()
        return self._row_to_dict(row) or {}

    def update_rl_job(
        self,
        job_id: int,
        *,
        status: str | None = None,
        progress: float | None = None,
        metrics: dict | None = None,
        error_message: str | None = None,
        model_id: int | None = None,
    ) -> dict | None:
        updates: list[str] = []
        values: list[Any] = []
        if status is not None:
            updates.append("status = ?")
            values.append(status)
        if progress is not None:
            updates.append("progress = ?")
            values.append(float(progress))
        if metrics is not None:
            updates.append("metrics_json = ?")
            values.append(json.dumps(metrics, sort_keys=True))
        if error_message is not None:
            updates.append("error_message = ?")
            values.append(error_message)
        if model_id is not None:
            updates.append("model_id = ?")
            values.append(int(model_id))
        updates.append("updated_at = ?")
        values.append(utcnow_iso())
        values.append(job_id)
        with self._lock:
            with self._connect() as conn:
                conn.execute(f"UPDATE rl_jobs SET {', '.join(updates)} WHERE id = ?", tuple(values))
                row = conn.execute("SELECT * FROM rl_jobs WHERE id = ?", (job_id,)).fetchone()
        return self._row_to_dict(row)

    def get_rl_job(self, job_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM rl_jobs WHERE id = ?", (job_id,)).fetchone()
        return self._row_to_dict(row)

    def list_rl_jobs(self, limit: int = 100) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM rl_jobs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [self._row_to_dict(row) or {} for row in rows]
