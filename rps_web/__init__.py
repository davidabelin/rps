"""Flask application factory and extension wiring for the RPS web app."""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask

from rps_storage.repository import RPSRepository
from rps_training.jobs import TrainingJobManager
from rps_web.runtime import GameRuntimeCache


def create_app(config: dict | None = None) -> Flask:
    """Create and configure the Flask application instance.

    Parameters
    ----------
    config : dict | None, optional
        Optional override map for tests/local customization.

    Returns
    -------
    Flask
        Fully configured Flask app with repositories, job managers, and
        blueprints registered.
    """

    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config.from_mapping(
        SECRET_KEY="dev-only-secret-key-change-me",
        DATABASE_URL=os.getenv("DATABASE_URL", ""),
        DB_PATH=os.getenv("DB_PATH", str(data_dir / "rps.db")),
        EVENTS_DIR=os.getenv("EVENTS_DIR", str(data_dir / "events")),
        MODELS_DIR=os.getenv("MODELS_DIR", str(data_dir / "models")),
        EXPORTS_DIR=os.getenv("EXPORTS_DIR", str(data_dir / "exports")),
        DEFAULT_AGENT="markov",
        TRAINING_EXECUTION_MODE=os.getenv("TRAINING_EXECUTION_MODE", "local"),
        TASKS_PROJECT_ID=os.getenv("TASKS_PROJECT_ID", ""),
        TASKS_LOCATION=os.getenv("TASKS_LOCATION", ""),
        TASKS_QUEUE=os.getenv("TASKS_QUEUE", ""),
        TRAINING_WORKER_URL=os.getenv("TRAINING_WORKER_URL", ""),
        TRAINING_WORKER_TOKEN=os.getenv("TRAINING_WORKER_TOKEN", ""),
        TRAINING_WORKER_SERVICE_ACCOUNT=os.getenv("TRAINING_WORKER_SERVICE_ACCOUNT", ""),
        INTERNAL_WORKER_TOKEN=os.getenv("INTERNAL_WORKER_TOKEN", ""),
        ROUND_EVENT_LOGGING_MODE=os.getenv("ROUND_EVENT_LOGGING_MODE", "auto"),
    )
    if config:
        app.config.update(config)

    database_target = app.config["DATABASE_URL"] or app.config["DB_PATH"]
    repository = RPSRepository(database_target)
    repository.init_schema()
    training_jobs = TrainingJobManager(
        repository,
        models_dir=app.config["MODELS_DIR"],
        execution_mode=app.config["TRAINING_EXECUTION_MODE"],
        task_project_id=app.config["TASKS_PROJECT_ID"] or None,
        task_location=app.config["TASKS_LOCATION"] or None,
        task_queue=app.config["TASKS_QUEUE"] or None,
        worker_url=app.config["TRAINING_WORKER_URL"] or None,
        worker_token=(app.config["TRAINING_WORKER_TOKEN"] or app.config["INTERNAL_WORKER_TOKEN"] or None),
        worker_service_account=app.config["TRAINING_WORKER_SERVICE_ACCOUNT"] or None,
    )
    app.extensions["repository"] = repository
    app.extensions["training_jobs"] = training_jobs
    app.extensions["game_runtime"] = GameRuntimeCache(max_entries=512)

    from rps_web.blueprints.game import game_bp
    from rps_web.blueprints.main import main_bp
    from rps_web.blueprints.benchmarks import benchmarks_bp
    from rps_web.blueprints.rl import rl_bp
    from rps_web.blueprints.training import training_bp
    from rps_rl.jobs import RLJobManager

    app.register_blueprint(main_bp)
    app.register_blueprint(game_bp)
    app.register_blueprint(training_bp)
    app.register_blueprint(benchmarks_bp)
    app.register_blueprint(rl_bp)
    app.extensions["rl_jobs"] = RLJobManager(repository, models_dir=app.config["MODELS_DIR"])
    return app
