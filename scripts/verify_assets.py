"""Verify authorized external assets before running the skill."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skill.executor import MountRideConfig, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/demo_handload_cross17.json"
    )
    args = parser.parse_args()
    config = MountRideConfig.load(ROOT / args.config)
    missing = []
    for key in config.raw["paths"]:
        path = config.resolve(key)
        if not path.is_file():
            missing.append({"key": key, "path": str(path.relative_to(ROOT))})
    if missing:
        print(
            json.dumps(
                {
                    "status": "missing_authorized_assets",
                    "missing": missing,
                    "notice": "See data/README.md and THIRD_PARTY_NOTICES.md.",
                },
                indent=2,
            )
        )
        raise SystemExit(2)
    checked = {}
    for relative, expected in config.raw["expected_sha256"].items():
        actual = sha256(ROOT / relative)
        if actual != expected:
            raise ValueError(f"asset hash mismatch: {relative}")
        checked[relative] = actual
    print(json.dumps({"status": "verified", "sha256": checked}, indent=2))


if __name__ == "__main__":
    main()
