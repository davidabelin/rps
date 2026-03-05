"""Asynchronous orchestration for RL training jobs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from rps_storage.repository import RPSRepository
from rps_storage.object_store import is_gcs_uri, join_storage_path
from rps_rl.trainer import RLTrainConfig, train_q_policy


class RLJobManager:
    """Manage RL job submission and lifecycle persistence."""

    def __init__(self, repository: RPSRepository, models_dir: str, max_workers: int = 1) -> None:
        """Initialize repository handle and local executor."""

        self.repository = repository
        self.models_dir = str(models_dir)
        if not is_gcs_uri(self.models_dir):
            Path(self.models_dir).mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="rps-rl")

    def submit_job(self, payload: dict) -> dict:
        """Persist and asynchronously start one RL training job."""

        config = self._config_from_payload(payload)
        job = self.repository.create_rl_job(payload)
        job_id = int(job["id"])
        self.executor.submit(self._run_job, job_id, config)
        return job

    @staticmethod
    def _config_from_payload(payload: dict) -> RLTrainConfig:
        """Normalize API payload into ``RLTrainConfig``."""

        opponents = payload.get("opponents")
        if opponents is None:
            opponents_tuple = RLTrainConfig().opponents
        elif isinstance(opponents, str):
            opponents_tuple = tuple(part.strip() for part in opponents.split(",") if part.strip())
        else:
            opponents_tuple = tuple(str(item) for item in opponents)
        return RLTrainConfig(
            episodes=int(payload.get("episodes", 300)),
            steps_per_episode=int(payload.get("steps_per_episode", 300)),
            alpha=float(payload.get("alpha", 0.15)),
            gamma=float(payload.get("gamma", 0.95)),
            epsilon_start=float(payload.get("epsilon_start", 1.0)),
            epsilon_end=float(payload.get("epsilon_end", 0.05)),
            epsilon_decay=float(payload.get("epsilon_decay", 0.995)),
            seed=int(payload.get("seed", 7)),
            opponents=opponents_tuple,
            opponent_schedule=str(payload.get("opponent_schedule", "curriculum")).strip() or "curriculum",
        )

    def _run_job(self, job_id: int, config: RLTrainConfig) -> None:
        """Execute end-to-end RL training lifecycle for one job id."""

        try:
            self.repository.update_rl_job(job_id, status="running", progress=0.05)
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            artifact_name = f"rl_qtable_job{job_id}_{timestamp}.pkl"
            artifact_path = join_storage_path(self.models_dir, artifact_name)
            self.repository.update_rl_job(job_id, progress=0.25)
            metrics = train_q_policy(config=config, artifact_path=artifact_path)
            self.repository.update_rl_job(job_id, progress=0.85, metrics=metrics)
            model_row = self.repository.create_model(
                name=f"rl-qtable-{timestamp}",
                model_type="rl_qtable",
                artifact_path=artifact_path,
                lookback=1,
                metrics=metrics,
            )
            self.repository.update_rl_job(
                job_id,
                status="completed",
                progress=1.0,
                metrics=metrics,
                model_id=int(model_row["id"]),
            )
        except Exception as exc:  # pragma: no cover
            self.repository.update_rl_job(job_id, status="failed", progress=1.0, error_message=str(exc))

    def shutdown(self) -> None:
        """Release local threadpool resources."""

        self.executor.shutdown(wait=False)
