"""CLI: execute the closed-loop mount-and-ride skill."""

import argparse
import json

from .executor import MountRideConfig, MountRideExecutor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/demo_handload_cross17.json"
    )
    parser.add_argument("--out", default="outputs/nominal")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--dr", choices=("none", "light"), default="none")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = MountRideConfig.load(args.config)
    executor = MountRideExecutor(
        config, dr_profile=args.dr, seed=args.seed
    )
    output = executor.run_to_directory(args.out, args.duration)
    summary = json.loads((output / "summary.json").read_text())
    print(
        json.dumps(
            {
                "output": str(output),
                "duration_s": summary["duration_s"],
                "native_ticks": summary["native_ticks"],
                "first_guard": summary["first_guard"],
                "crossing_completed": summary["crossing_completed"],
                "crossing_complete_s": summary["crossing_complete_s"],
                "dr_profile": summary["dr_profile"],
                "seed": summary["seed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
