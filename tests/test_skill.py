import json
from pathlib import Path
import unittest

import numpy as np

from skill.executor import MountRideConfig, MountRideExecutor
from skill.phases import CrossingThresholds, CrossingTracker


ROOT = Path(__file__).resolve().parents[1]
CONFIG = MountRideConfig.load(
    ROOT / "configs/demo_handload_cross17.json"
)
RESTRICTED_ASSETS_AVAILABLE = all(
    CONFIG.resolve(key).is_file()
    for key in (
        "model",
        "actor",
        "reference",
        "initial_state",
        "contract",
        "validated_trajectory",
    )
)


class SkillTests(unittest.TestCase):
    @unittest.skipUnless(
        RESTRICTED_ASSETS_AVAILABLE,
        "authorized model/policy/reference assets are not distributed",
    )
    def test_assets_and_contract(self):
        CONFIG.verify_assets()
        with np.load(
            CONFIG.resolve("validated_trajectory"), allow_pickle=False
        ) as trajectory:
            self.assertEqual(trajectory["qpos"].shape, (17001, 50))
            self.assertEqual(trajectory["qvel"].shape, (17001, 48))

    def test_crossing_phase(self):
        tracker = CrossingTracker(CrossingThresholds())
        self.assertEqual(tracker.update([-0.15, 0.2, 0.0], 0.0), "crossing")
        self.assertEqual(
            tracker.update([-0.20, -0.10, 0.0], 1.0),
            "post_cross_support_hold",
        )
        self.assertTrue(tracker.crossed)

    def test_light_dr_candidate_result_is_rejected(self):
        report = (
            ROOT / "training/RESULTS_light_dr_ppo16.md"
        ).read_text()
        done = json.loads(
            (ROOT / "docs/light_dr_ppo16/DONE.json").read_text()
        )
        self.assertIn("Rejected, no promotion", report)
        self.assertEqual(done["accepted_updates"], 16)
        self.assertIsNone(done["formal_success"])

    @unittest.skipUnless(
        RESTRICTED_ASSETS_AVAILABLE,
        "authorized model/policy/reference assets are not distributed",
    )
    def test_closed_loop_reproduces_source_prefix(self):
        result = MountRideExecutor(CONFIG).run(duration_s=0.1)
        with np.load(
            CONFIG.resolve("validated_trajectory"), allow_pickle=False
        ) as source:
            expected = source["qpos"][1:101]
        maximum_error = float(
            np.max(np.abs(result["qpos"] - expected))
        )
        self.assertLess(maximum_error, 5e-7)


if __name__ == "__main__":
    unittest.main()
