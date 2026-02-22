from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rps_agents import build_heuristic_agent, list_agent_specs
from rps_core.simulator import leaderboard, run_round_robin


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a round-robin tournament across heuristic agents.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes per matchup")
    parser.add_argument("--steps", type=int, default=300, help="Rounds per episode")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--agents", type=str, default="", help="Comma-separated agent names")
    parser.add_argument("--output", type=str, default="data/exports", help="Output directory")
    args = parser.parse_args()

    available = {spec.name: spec for spec in list_agent_specs()}
    if args.agents.strip():
        selected_names = [name.strip() for name in args.agents.split(",") if name.strip()]
    else:
        selected_names = sorted(available.keys())
    missing = [name for name in selected_names if name not in available]
    if missing:
        raise SystemExit(f"Unknown agents requested: {', '.join(missing)}")

    factories = {name: (lambda n=name: build_heuristic_agent(n)) for name in selected_names}
    rows = run_round_robin(factories, episodes=args.episodes, steps=args.steps, seed=args.seed)
    board = leaderboard(rows)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = output_dir / f"tournament_{timestamp}.json"
    out_path.write_text(json.dumps({"matchups": rows, "leaderboard": board}, indent=2), encoding="utf-8")

    print("Leaderboard:")
    for item in board:
        print(
            f"- {item['agent']}: win={item['avg_win_rate']:.3f} "
            f"loss={item['avg_loss_rate']:.3f} draw={item['avg_draw_rate']:.3f}"
        )
    print(f"\nSaved report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
