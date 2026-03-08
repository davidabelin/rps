"""Asynchronous orchestration for persisted agent-vs-agent RPS matches."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from rps_agents import ModelBackedAgent, build_heuristic_agent, list_agent_specs
from rps_core.matches import play_agent_match
from rps_storage.repository import RPSRepository


def _available_agent_names() -> list[str]:
    return [spec.name for spec in list_agent_specs()]


def _default_match_opponent(agent_name: str) -> str:
    for candidate in _available_agent_names():
        if candidate != agent_name:
            return candidate
    return agent_name


def _build_agent_from_name(repository: RPSRepository, agent_name: str):
    if agent_name == "active_model":
        model_record = repository.get_active_model()
        if model_record is None:
            raise RuntimeError("No active model is available. Activate a trained model first.")
        return ModelBackedAgent(str(model_record["artifact_path"]))
    if agent_name not in set(_available_agent_names()):
        raise KeyError(f"Unknown agent '{agent_name}'.")
    return build_heuristic_agent(agent_name)


class MatchJobManager:
    """Manage background arena matches and persisted replay traces."""

    def __init__(self, repository: RPSRepository, default_agent: str, max_workers: int = 2) -> None:
        self.repository = repository
        self.default_agent = str(default_agent)
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="rps-arena")

    def submit_job(self, payload: dict) -> dict:
        config = self._config_from_payload(payload)
        _build_agent_from_name(self.repository, config["agent_a"])
        _build_agent_from_name(self.repository, config["agent_b"])
        job = self.repository.create_arena_match(
            agent_a_name=config["agent_a"],
            agent_b_name=config["agent_b"],
            params=config,
        )
        self.executor.submit(self._run_job, int(job["id"]), config)
        return job

    def _config_from_payload(self, payload: dict) -> dict:
        agent_a = str(payload.get("agent_a", self.default_agent))
        agent_b = str(payload.get("agent_b", _default_match_opponent(agent_a)))
        rounds = int(payload.get("rounds", 50))
        if rounds <= 0 or rounds > 5000:
            raise ValueError("rounds must be between 1 and 5000.")
        raw_seed = payload.get("seed")
        seed = int(raw_seed) if raw_seed is not None else None
        return {
            "agent_a": agent_a,
            "agent_b": agent_b,
            "rounds": rounds,
            "seed": seed,
        }

    def _run_job(self, job_id: int, config: dict) -> None:
        trace: list[dict] = []
        rounds = int(config["rounds"])
        try:
            self.repository.update_arena_match(
                job_id,
                status="running",
                progress=0.01,
                summary={
                    "mode": "agent_vs_agent",
                    "agent_a": config["agent_a"],
                    "agent_b": config["agent_b"],
                    "rounds": rounds,
                    "seed": config["seed"],
                    "score_agent_a": 0,
                    "score_agent_b": 0,
                    "score_ties": 0,
                    "winner": None,
                },
                trace=[],
            )

            agent_a = _build_agent_from_name(self.repository, config["agent_a"])
            agent_b = _build_agent_from_name(self.repository, config["agent_b"])

            def _on_round(frame: dict) -> None:
                trace.append(frame)
                self.repository.update_arena_match(
                    job_id,
                    status="running",
                    progress=len(trace) / max(1, rounds),
                    summary={
                        "mode": "agent_vs_agent",
                        "agent_a": config["agent_a"],
                        "agent_b": config["agent_b"],
                        "rounds": rounds,
                        "seed": config["seed"],
                        "score_agent_a": int(frame["score_agent_a"]),
                        "score_agent_b": int(frame["score_agent_b"]),
                        "score_ties": int(frame["score_ties"]),
                        "winner": None,
                    },
                    trace=list(trace),
                )

            match = play_agent_match(
                agent_a=agent_a,
                agent_b=agent_b,
                agent_a_name=config["agent_a"],
                agent_b_name=config["agent_b"],
                rounds=rounds,
                seed=config["seed"],
                on_round=_on_round,
            )
            summary = {key: value for key, value in match.items() if key != "trace"}
            self.repository.update_arena_match(
                job_id,
                status="completed",
                progress=1.0,
                winner=str(match["winner"]),
                summary=summary,
                trace=match["trace"],
            )
        except Exception as exc:  # pragma: no cover
            self.repository.update_arena_match(
                job_id,
                status="failed",
                progress=1.0,
                error_message=str(exc),
                trace=trace,
            )

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False)
