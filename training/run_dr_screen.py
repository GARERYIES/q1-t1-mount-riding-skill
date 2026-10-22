"""Run a bounded nominal/light-DR native screen without PPO promotion."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skill.executor import MountRideConfig, MountRideExecutor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/demo_handload_cross17.json"
    )
    parser.add_argument("--out", default="outputs/dr_screen")
    parser.add_argument("--duration", type=float, default=17.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()
    config = MountRideConfig.load(args.config)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    for profile in ("none", "light"):
        seeds = [0] if profile == "none" else args.seeds
        for seed in seeds:
            case_dir = output / f"{profile}_seed{seed}"
            MountRideExecutor(
                config,
                dr_profile=profile,
                seed=seed,
            ).run_to_directory(case_dir, duration_s=args.duration)
            summary = json.loads((case_dir / "summary.json").read_text())
            final = summary["samples"][-1]
            cases.append(
                {
                    "profile": profile,
                    "seed": seed,
                    "duration_s": summary["duration_s"],
                    "first_guard": summary["first_guard"],
                    "crossing_completed": summary["crossing_completed"],
                    "crossing_complete_s": summary["crossing_complete_s"],
                    "final": final,
                }
            )
    result = {
        "schema": "q1-t1-light-dr-screen-v1",
        "scope": "Nominal plus bounded light-randomization candidate screen.",
        "duration_per_case_s": args.duration,
        "cases": cases,
        "promotion": None,
    }
    (output / "screen_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
