"""CLI entry point for canonical benchmark evaluations."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rps_agents import ModelBackedAgent, build_heuristic_agent, list_agent_specs
from rps_benchmarks import benchmark_agent
from rps_storage import RPSRepository


def resolve_agent(agent_name: str, repository: RPSRepository):
    """Resolve evaluated agent factory by CLI name."""

    available = {spec.name for spec in list_agent_specs()}
    if agent_name == "active_model":
        model = repository.get_active_model()
        if model is None:
            raise SystemExit("No active model is set. Activate one first.")
        artifact = model["artifact_path"]
        return lambda: ModelBackedAgent(artifact)
    if agent_name not in available:
        raise SystemExit(f"Unknown agent: {agent_name}")
    return lambda: build_heuristic_agent(agent_name)


def main() -> int:
    """Run benchmark CLI and write JSON report."""

    parser = argparse.ArgumentParser(description="Benchmark an agent against canonical RPS bots.")
    parser.add_argument("--agent", type=str, default="markov")
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--db-path", type=str, default="data/rps.db")
    parser.add_argument("--output", type=str, default="data/exports")
    args = parser.parse_args()

    repository = RPSRepository(args.db_path)
    repository.init_schema()
    factory = resolve_agent(args.agent, repository)
    result = benchmark_agent(factory, rounds=args.rounds, seed=args.seed)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = output_dir / f"benchmark_{args.agent}_{timestamp}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nSaved benchmark report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
