"""Validate and summarize one closed-loop skill run."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    summary = json.loads((run / "summary.json").read_text())
    with np.load(run / "trajectory.npz", allow_pickle=False) as trajectory:
        time = trajectory["time"]
        qpos = trajectory["qpos"]
        qvel = trajectory["qvel"]
    if len(time) != summary["native_ticks"]:
        raise ValueError("summary/trajectory tick mismatch")
    if not np.all(np.diff(time) > 0):
        raise ValueError("non-increasing native clock")
    if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        raise ValueError("nonfinite trajectory")
    if summary["external_forces"] or summary["root_fixed"]:
        raise ValueError("run used a forbidden support mechanism")
    if summary["mid_episode_reset"]:
        raise ValueError("run used a mid-episode reset")
    last = summary["samples"][-1]
    report = {
        "run": str(run),
        "duration_s": summary["duration_s"],
        "native_ticks": summary["native_ticks"],
        "start_type": summary["start_type"],
        "crossing_completed": summary["crossing_completed"],
        "crossing_complete_s": summary["crossing_complete_s"],
        "first_guard": summary["first_guard"],
        "final": {
            "left_palm_Fz_N": last["left_palm_Fz_N"],
            "right_palm_Fz_N": last["right_palm_Fz_N"],
            "seat_Fz_N": last["seat_Fz_N"],
            "left_foot_Fz_N": last["left_foot_Fz_N"],
            "right_foot_Fz_N": last["right_foot_Fz_N"],
            "unapproved_T1_Fn_N": last["unapproved_T1_Fn_N"],
            "root_height_m": last["root_height_m"],
            "T1_tilt_deg": last["T1_tilt_deg"],
        },
        "formal_legal_hold_claimed": False,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
