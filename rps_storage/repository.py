"""Repository layer for app metadata using SQLite or PostgreSQL via SQLAlchemy.

The repository exposes dict-based CRUD helpers consumed by web routes and job
managers. It supports:

- local SQLite paths for development/tests
- SQL database URLs (Cloud SQL PostgreSQL in production)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, MappingResult
from sqlalchemy.exc import DatabaseError


def utcnow_iso() -> str:
    """Return timezone-aware UTC timestamp string."""

    return datetime.now(UTC).isoformat()


def _looks_like_database_url(value: str) -> bool:
    """Best-effort check for SQLAlchemy-style DB URLs."""

    return "://" in str(value)


def _to_sqlite_url(path_value: str) -> str:
    """Convert filesystem path into SQLAlchemy SQLite URL."""

    path = Path(path_value).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite+pysqlite:///{path.as_posix()}"


class RPSRepository:
    """Persistence facade for games, rounds, jobs, and models tables.

    Parameters
    ----------
    db_target : str
        Local SQLite path or SQLAlchemy URL.
    """

    def __init__(self, db_target: str) -> None:
        """Initialize engine, dialect metadata, and synchronization lock."""

        self._lock = RLock()
        target = str(db_target).strip()
        if not target:
            raise ValueError("Database target must not be empty.")
        self.db_target = target
        self.db_url = target if _looks_like_database_url(target) else _to_sqlite_url(target)
        connect_args = {"check_same_thread": False} if self.db_url.startswith("sqlite") else {}
        self.engine: Engine = create_engine(self.db_url, future=True, pool_pre_ping=True, connect_args=connect_args)
        self._dialect = self.engine.dialect.name

    def _run_script(self, script: str) -> None:
        """Execute semicolon-delimited SQL script statement-by-statement."""

        statements = [part.strip() for part in script.split(";") if part.strip()]
        with self.engine.begin() as conn:
            for statement in statements:
                conn.execute(text(statement))

    def init_schema(self) -> None:
        """Create required tables/indexes for the active dialect."""

        sqlite_schema = """
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

        postgres_schema = """
            CREATE TABLE IF NOT EXISTS games (
                id BIGSERIAL PRIMARY KEY,
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
                id BIGSERIAL PRIMARY KEY,
                game_id BIGINT NOT NULL REFERENCES games(id),
                session_index INTEGER NOT NULL,
                round_index INTEGER NOT NULL,
                player_action INTEGER NOT NULL,
                ai_action INTEGER NOT NULL,
                outcome TEXT NOT NULL,
                reward_delta INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS training_jobs (
                id BIGSERIAL PRIMARY KEY,
                status TEXT NOT NULL,
                model_type TEXT NOT NULL,
                params_json TEXT NOT NULL,
                progress DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                metrics_json TEXT,
                error_message TEXT,
                model_id BIGINT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS models (
                id BIGSERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                model_type TEXT NOT NULL,
                artifact_path TEXT NOT NULL,
                lookback INTEGER NOT NULL,
                metrics_json TEXT,
                is_active INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS model_metrics (
                id BIGSERIAL PRIMARY KEY,
                model_id BIGINT NOT NULL REFERENCES models(id),
                metric_name TEXT NOT NULL,
                metric_value DOUBLE PRECISION NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS rl_jobs (
                id BIGSERIAL PRIMARY KEY,
                status TEXT NOT NULL,
                params_json TEXT NOT NULL,
                progress DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                metrics_json TEXT,
                error_message TEXT,
                model_id BIGINT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_rounds_game_session ON rounds(game_id, session_index, round_index);
            CREATE INDEX IF NOT EXISTS idx_models_active ON models(is_active);
            CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON training_jobs(status);
            CREATE INDEX IF NOT EXISTS idx_rl_jobs_status ON rl_jobs(status);
        """

        with self._lock:
            self._run_script(sqlite_schema if self._dialect == "sqlite" else postgres_schema)

    @staticmethod
    def _first_or_none(rows: MappingResult) -> dict | None:
        """Return first mapping row as ``dict`` or ``None``."""

        row = rows.first()
        if row is None:
            return None
        return dict(row)

    def _insert_and_fetch(self, insert_sql: str, params: dict[str, Any], table_name: str) -> dict:
        """Insert one row and fetch inserted record across dialects.

        Notes
        -----
        Uses ``RETURNING`` when available, with safe fallbacks for SQLite.
        """

        returning_sql = f"{insert_sql} RETURNING *"
        with self._lock:
            with self.engine.begin() as conn:
                try:
                    row = self._first_or_none(conn.execute(text(returning_sql), params).mappings())
                    if row is not None:
                        return row
                except DatabaseError:
                    pass
                result = conn.execute(text(insert_sql), params)
                last_id = getattr(result, "lastrowid", None)
                if last_id is not None:
                    row = self._first_or_none(
                        conn.execute(text(f"SELECT * FROM {table_name} WHERE id = :id"), {"id": int(last_id)}).mappings()
                    )
                    if row is not None:
                        return row
                row = self._first_or_none(conn.execute(text(f"SELECT * FROM {table_name} ORDER BY id DESC LIMIT 1")).mappings())
                return row or {}

    def create_game(self, agent_name: str) -> dict:
        """Create a new game row initialized to zero score."""

        now = utcnow_iso()
        return self._insert_and_fetch(
            """
            INSERT INTO games (agent_name, session_index, rounds_played, score_player, score_ai, score_ties, created_at, updated_at)
            VALUES (:agent_name, 0, 0, 0, 0, 0, :created_at, :updated_at)
            """,
            {"agent_name": agent_name, "created_at": now, "updated_at": now},
            "games",
        )

    def get_game(self, game_id: int) -> dict | None:
        """Fetch one game row by id."""

        with self.engine.begin() as conn:
            return self._first_or_none(conn.execute(text("SELECT * FROM games WHERE id = :id"), {"id": game_id}).mappings())

    def update_game_scores(
        self,
        game_id: int,
        rounds_played: int,
        score_player: int,
        score_ai: int,
        score_ties: int,
    ) -> dict:
        """Update rolling game score counters."""

        now = utcnow_iso()
        with self._lock:
            with self.engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        UPDATE games
                        SET rounds_played = :rounds_played, score_player = :score_player,
                            score_ai = :score_ai, score_ties = :score_ties, updated_at = :updated_at
                        WHERE id = :id
                        """
                    ),
                    {
                        "rounds_played": rounds_played,
                        "score_player": score_player,
                        "score_ai": score_ai,
                        "score_ties": score_ties,
                        "updated_at": now,
                        "id": game_id,
                    },
                )
                row = self._first_or_none(conn.execute(text("SELECT * FROM games WHERE id = :id"), {"id": game_id}).mappings())
        return row or {}

    def persist_game_scores(
        self,
        game_id: int,
        rounds_played: int,
        score_player: int,
        score_ai: int,
        score_ties: int,
    ) -> str:
        """Persist rolling game score counters without fetching row back.

        Returns
        -------
        str
            Timestamp string written to ``updated_at``.
        """

        now = utcnow_iso()
        with self._lock:
            with self.engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        UPDATE games
                        SET rounds_played = :rounds_played, score_player = :score_player,
                            score_ai = :score_ai, score_ties = :score_ties, updated_at = :updated_at
                        WHERE id = :id
                        """
                    ),
                    {
                        "rounds_played": rounds_played,
                        "score_player": score_player,
                        "score_ai": score_ai,
                        "score_ties": score_ties,
                        "updated_at": now,
                        "id": game_id,
                    },
                )
        return now

    def reset_game(self, game_id: int) -> dict | None:
        """Start a new session for an existing game id."""

        game = self.get_game(game_id)
        if not game:
            return None
        now = utcnow_iso()
        new_session_index = int(game["session_index"]) + 1
        with self._lock:
            with self.engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        UPDATE games
                        SET session_index = :session_index, rounds_played = 0, score_player = 0,
                            score_ai = 0, score_ties = 0, updated_at = :updated_at
                        WHERE id = :id
                        """
                    ),
                    {"session_index": new_session_index, "updated_at": now, "id": game_id},
                )
                row = self._first_or_none(conn.execute(text("SELECT * FROM games WHERE id = :id"), {"id": game_id}).mappings())
        return row

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
        """Insert one persisted round row."""

        now = utcnow_iso()
        return self._insert_and_fetch(
            """
            INSERT INTO rounds
            (game_id, session_index, round_index, player_action, ai_action, outcome, reward_delta, created_at)
            VALUES (:game_id, :session_index, :round_index, :player_action, :ai_action, :outcome, :reward_delta, :created_at)
            """,
            {
                "game_id": game_id,
                "session_index": session_index,
                "round_index": round_index,
                "player_action": player_action,
                "ai_action": ai_action,
                "outcome": outcome,
                "reward_delta": reward_delta,
                "created_at": now,
            },
            "rounds",
        )

    def record_round_and_update_game(
        self,
        *,
        game_id: int,
        session_index: int,
        round_index: int,
        player_action: int,
        ai_action: int,
        outcome: str,
        reward_delta: int,
    ) -> tuple[dict, dict]:
        """Persist one round and increment game scores in one transaction.

        Parameters
        ----------
        game_id : int
            Target game id.
        session_index : int
            Expected active session index for optimistic safety.
        round_index : int
            Zero-based round index within the active session.
        player_action : int
            Human action id.
        ai_action : int
            Agent action id.
        outcome : str
            Outcome label from player perspective.
        reward_delta : int
            Reward sign from player perspective.

        Returns
        -------
        tuple[dict, dict]
            ``(stored_round, updated_game)`` payload.

        Raises
        ------
        KeyError
            If game does not exist or session index no longer matches.
        """

        now = utcnow_iso()
        score_player_inc = 1 if int(reward_delta) > 0 else 0
        score_ai_inc = 1 if int(reward_delta) < 0 else 0
        score_ties_inc = 1 if int(reward_delta) == 0 else 0

        round_params = {
            "game_id": int(game_id),
            "session_index": int(session_index),
            "round_index": int(round_index),
            "player_action": int(player_action),
            "ai_action": int(ai_action),
            "outcome": str(outcome),
            "reward_delta": int(reward_delta),
            "created_at": now,
        }
        game_params = {
            "id": int(game_id),
            "session_index": int(session_index),
            "score_player_inc": int(score_player_inc),
            "score_ai_inc": int(score_ai_inc),
            "score_ties_inc": int(score_ties_inc),
            "updated_at": now,
        }

        with self._lock:
            with self.engine.begin() as conn:
                stored_round = None
                try:
                    stored_round = self._first_or_none(
                        conn.execute(
                            text(
                                """
                                INSERT INTO rounds
                                (game_id, session_index, round_index, player_action, ai_action, outcome, reward_delta, created_at)
                                VALUES (:game_id, :session_index, :round_index, :player_action, :ai_action, :outcome, :reward_delta, :created_at)
                                RETURNING *
                                """
                            ),
                            round_params,
                        ).mappings()
                    )
                except DatabaseError:
                    stored_round = None
                if stored_round is None:
                    insert_result = conn.execute(
                        text(
                            """
                            INSERT INTO rounds
                            (game_id, session_index, round_index, player_action, ai_action, outcome, reward_delta, created_at)
                            VALUES (:game_id, :session_index, :round_index, :player_action, :ai_action, :outcome, :reward_delta, :created_at)
                            """
                        ),
                        round_params,
                    )
                    last_id = getattr(insert_result, "lastrowid", None)
                    if last_id is None:
                        stored_round = self._first_or_none(
                            conn.execute(text("SELECT * FROM rounds ORDER BY id DESC LIMIT 1")).mappings()
                        )
                    else:
                        stored_round = self._first_or_none(
                            conn.execute(text("SELECT * FROM rounds WHERE id = :id"), {"id": int(last_id)}).mappings()
                        )

                updated_game = None
                try:
                    updated_game = self._first_or_none(
                        conn.execute(
                            text(
                                """
                                UPDATE games
                                SET rounds_played = rounds_played + 1,
                                    score_player = score_player + :score_player_inc,
                                    score_ai = score_ai + :score_ai_inc,
                                    score_ties = score_ties + :score_ties_inc,
                                    updated_at = :updated_at
                                WHERE id = :id AND session_index = :session_index
                                RETURNING *
                                """
                            ),
                            game_params,
                        ).mappings()
                    )
                except DatabaseError:
                    updated_game = None
                if updated_game is None:
                    update_result = conn.execute(
                        text(
                            """
                            UPDATE games
                            SET rounds_played = rounds_played + 1,
                                score_player = score_player + :score_player_inc,
                                score_ai = score_ai + :score_ai_inc,
                                score_ties = score_ties + :score_ties_inc,
                                updated_at = :updated_at
                            WHERE id = :id AND session_index = :session_index
                            """
                        ),
                        game_params,
                    )
                    if getattr(update_result, "rowcount", 0) == 0:
                        raise KeyError(f"Game {game_id} not found or session mismatch.")
                    updated_game = self._first_or_none(
                        conn.execute(text("SELECT * FROM games WHERE id = :id"), {"id": int(game_id)}).mappings()
                    )

                if updated_game is None:
                    raise KeyError(f"Game {game_id} not found or session mismatch.")
                return stored_round or {}, updated_game

    def list_rounds(self, game_id: int, session_index: int | None = None) -> list[dict]:
        """List rounds for a game, optionally constrained to one session."""

        query = """
            SELECT * FROM rounds
            WHERE game_id = :game_id
        """
        params: dict[str, Any] = {"game_id": game_id}
        if session_index is not None:
            query += " AND session_index = :session_index"
            params["session_index"] = session_index
        query += " ORDER BY round_index ASC, id ASC"
        with self.engine.begin() as conn:
            rows = conn.execute(text(query), params).mappings().all()
        return [dict(row) for row in rows]

    def list_rounds_for_training(self) -> list[dict]:
        """List all rounds in deterministic order for dataset building."""

        with self.engine.begin() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT game_id, session_index, round_index, player_action, ai_action, outcome, reward_delta, created_at
                    FROM rounds
                    ORDER BY game_id ASC, session_index ASC, round_index ASC
                    """
                )
            ).mappings().all()
        return [dict(row) for row in rows]

    def create_training_job(self, model_type: str, params: dict) -> dict:
        """Create queued supervised training job record."""

        now = utcnow_iso()
        return self._insert_and_fetch(
            """
            INSERT INTO training_jobs
            (status, model_type, params_json, progress, metrics_json, error_message, model_id, created_at, updated_at)
            VALUES ('queued', :model_type, :params_json, 0.0, NULL, NULL, NULL, :created_at, :updated_at)
            """,
            {"model_type": model_type, "params_json": json.dumps(params, sort_keys=True), "created_at": now, "updated_at": now},
            "training_jobs",
        )

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
        """Patch mutable supervised-job fields and return latest row."""

        updates: list[str] = []
        params: dict[str, Any] = {}
        if status is not None:
            updates.append("status = :status")
            params["status"] = status
        if progress is not None:
            updates.append("progress = :progress")
            params["progress"] = float(progress)
        if metrics is not None:
            updates.append("metrics_json = :metrics_json")
            params["metrics_json"] = json.dumps(metrics, sort_keys=True)
        if error_message is not None:
            updates.append("error_message = :error_message")
            params["error_message"] = error_message
        if model_id is not None:
            updates.append("model_id = :model_id")
            params["model_id"] = int(model_id)
        updates.append("updated_at = :updated_at")
        params["updated_at"] = utcnow_iso()
        params["id"] = job_id
        with self._lock:
            with self.engine.begin() as conn:
                conn.execute(text(f"UPDATE training_jobs SET {', '.join(updates)} WHERE id = :id"), params)
                row = self._first_or_none(
                    conn.execute(text("SELECT * FROM training_jobs WHERE id = :id"), {"id": job_id}).mappings()
                )
        return row

    def get_training_job(self, job_id: int) -> dict | None:
        """Fetch supervised training job by id."""

        with self.engine.begin() as conn:
            return self._first_or_none(
                conn.execute(text("SELECT * FROM training_jobs WHERE id = :id"), {"id": job_id}).mappings()
            )

    def list_training_jobs(self, limit: int = 100) -> list[dict]:
        """List recent supervised training jobs (descending id)."""

        with self.engine.begin() as conn:
            rows = conn.execute(text("SELECT * FROM training_jobs ORDER BY id DESC LIMIT :limit"), {"limit": limit}).mappings().all()
        return [dict(row) for row in rows]

    def create_model(self, name: str, model_type: str, artifact_path: str, lookback: int, metrics: dict) -> dict:
        """Create model metadata row and numeric metric rows.

        Parameters
        ----------
        name : str
            Display/version name.
        model_type : str
            Model family identifier.
        artifact_path : str
            Local/GCS artifact location.
        lookback : int
            Feature lookback window used for this model.
        metrics : dict
            Arbitrary metrics payload persisted as JSON; numeric values are also
            split into ``model_metrics`` rows.
        """

        now = utcnow_iso()
        with self._lock:
            with self.engine.begin() as conn:
                row = None
                try:
                    row = self._first_or_none(
                        conn.execute(
                            text(
                                """
                                INSERT INTO models (name, model_type, artifact_path, lookback, metrics_json, is_active, created_at)
                                VALUES (:name, :model_type, :artifact_path, :lookback, :metrics_json, 0, :created_at)
                                RETURNING *
                                """
                            ),
                            {
                                "name": name,
                                "model_type": model_type,
                                "artifact_path": artifact_path,
                                "lookback": lookback,
                                "metrics_json": json.dumps(metrics, sort_keys=True),
                                "created_at": now,
                            },
                        ).mappings()
                    )
                except DatabaseError:
                    row = None
                if row is None:
                    result = conn.execute(
                        text(
                            """
                            INSERT INTO models (name, model_type, artifact_path, lookback, metrics_json, is_active, created_at)
                            VALUES (:name, :model_type, :artifact_path, :lookback, :metrics_json, 0, :created_at)
                            """
                        ),
                        {
                            "name": name,
                            "model_type": model_type,
                            "artifact_path": artifact_path,
                            "lookback": lookback,
                            "metrics_json": json.dumps(metrics, sort_keys=True),
                            "created_at": now,
                        },
                    )
                    model_id = int(getattr(result, "lastrowid", 0))
                    if model_id <= 0:
                        latest = self._first_or_none(conn.execute(text("SELECT * FROM models ORDER BY id DESC LIMIT 1")).mappings())
                        model_id = int(latest["id"]) if latest else 0
                    row = self._first_or_none(conn.execute(text("SELECT * FROM models WHERE id = :id"), {"id": model_id}).mappings())
                model_id = int(row["id"]) if row else 0
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        conn.execute(
                            text(
                                """
                                INSERT INTO model_metrics (model_id, metric_name, metric_value, created_at)
                                VALUES (:model_id, :metric_name, :metric_value, :created_at)
                                """
                            ),
                            {
                                "model_id": model_id,
                                "metric_name": key,
                                "metric_value": float(value),
                                "created_at": now,
                            },
                        )
        return row or {}

    def list_models(self, limit: int = 200) -> list[dict]:
        """List recent model records (descending id)."""

        with self.engine.begin() as conn:
            rows = conn.execute(text("SELECT * FROM models ORDER BY id DESC LIMIT :limit"), {"limit": limit}).mappings().all()
        return [dict(row) for row in rows]

    def get_model(self, model_id: int) -> dict | None:
        """Fetch one model row by id."""

        with self.engine.begin() as conn:
            return self._first_or_none(conn.execute(text("SELECT * FROM models WHERE id = :id"), {"id": model_id}).mappings())

    def get_active_model(self) -> dict | None:
        """Fetch currently active model row (latest active if multiple)."""

        with self.engine.begin() as conn:
            return self._first_or_none(
                conn.execute(text("SELECT * FROM models WHERE is_active = 1 ORDER BY id DESC LIMIT 1")).mappings()
            )

    def activate_model(self, model_id: int) -> dict | None:
        """Deactivate all models and activate one target model id."""

        with self._lock:
            with self.engine.begin() as conn:
                conn.execute(text("UPDATE models SET is_active = 0"))
                conn.execute(text("UPDATE models SET is_active = 1 WHERE id = :id"), {"id": model_id})
                row = self._first_or_none(conn.execute(text("SELECT * FROM models WHERE id = :id"), {"id": model_id}).mappings())
        return row

    def create_rl_job(self, params: dict) -> dict:
        """Create queued RL training job record."""

        now = utcnow_iso()
        return self._insert_and_fetch(
            """
            INSERT INTO rl_jobs
            (status, params_json, progress, metrics_json, error_message, model_id, created_at, updated_at)
            VALUES ('queued', :params_json, 0.0, NULL, NULL, NULL, :created_at, :updated_at)
            """,
            {"params_json": json.dumps(params, sort_keys=True), "created_at": now, "updated_at": now},
            "rl_jobs",
        )

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
        """Patch mutable RL-job fields and return latest row."""

        updates: list[str] = []
        named: dict[str, Any] = {}
        if status is not None:
            updates.append("status = :status")
            named["status"] = status
        if progress is not None:
            updates.append("progress = :progress")
            named["progress"] = float(progress)
        if metrics is not None:
            updates.append("metrics_json = :metrics_json")
            named["metrics_json"] = json.dumps(metrics, sort_keys=True)
        if error_message is not None:
            updates.append("error_message = :error_message")
            named["error_message"] = error_message
        if model_id is not None:
            updates.append("model_id = :model_id")
            named["model_id"] = int(model_id)
        updates.append("updated_at = :updated_at")
        named["updated_at"] = utcnow_iso()
        named["id"] = job_id
        with self._lock:
            with self.engine.begin() as conn:
                conn.execute(text(f"UPDATE rl_jobs SET {', '.join(updates)} WHERE id = :id"), named)
                row = self._first_or_none(conn.execute(text("SELECT * FROM rl_jobs WHERE id = :id"), {"id": job_id}).mappings())
        return row

    def get_rl_job(self, job_id: int) -> dict | None:
        """Fetch RL job by id."""

        with self.engine.begin() as conn:
            return self._first_or_none(conn.execute(text("SELECT * FROM rl_jobs WHERE id = :id"), {"id": job_id}).mappings())

    def list_rl_jobs(self, limit: int = 100) -> list[dict]:
        """List recent RL jobs (descending id)."""

        with self.engine.begin() as conn:
            rows = conn.execute(text("SELECT * FROM rl_jobs ORDER BY id DESC LIMIT :limit"), {"limit": limit}).mappings().all()
        return [dict(row) for row in rows]
