"""Benchmark API routes for canonical-bot evaluations."""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request

from rps_agents import ModelBackedAgent, build_heuristic_agent, list_agent_specs
from rps_benchmarks import benchmark_agent

benchmarks_bp = Blueprint("benchmarks_api", __name__, url_prefix="/api/v1/benchmarks")


def _repo():
    """Return storage repository extension."""

    return current_app.extensions["repository"]


def _resolve_agent_factory(agent_name: str):
    """Resolve benchmark target agent factory from request name."""

    available = {spec.name for spec in list_agent_specs()}
    if agent_name == "active_model":
        model = _repo().get_active_model()
        if model is None:
            raise ValueError("No active model is available. Train and activate one first.")
        artifact_path = model["artifact_path"]
        return lambda: ModelBackedAgent(artifact_path)
    if agent_name not in available:
        raise ValueError(f"Unknown agent '{agent_name}'.")
    return lambda: build_heuristic_agent(agent_name)


@benchmarks_bp.post("/run")
def run_benchmark():
    """Run benchmark against canonical bots for selected agent."""

    payload = request.get_json(silent=True) or {}
    agent_name = str(payload.get("agent", "markov"))
    rounds = int(payload.get("rounds", 1000))
    seed = int(payload.get("seed", 7))
    if rounds < 50:
        return jsonify({"error": "rounds must be at least 50 for meaningful benchmark results."}), 400
    try:
        agent_factory = _resolve_agent_factory(agent_name)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    try:
        results = benchmark_agent(agent_factory=agent_factory, rounds=rounds, seed=seed)
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": f"Benchmark failed: {exc}"}), 500
    return jsonify({"agent": agent_name, "benchmark": results})
