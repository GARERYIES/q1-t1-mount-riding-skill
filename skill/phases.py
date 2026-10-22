"""Stable skill phase names and crossing detector."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CrossingThresholds:
    seen_near_y_m: float = 0.05
    crossed_y_m: float = -0.05
    crossed_x_max_m: float = -0.18


class CrossingTracker:
    def __init__(self, thresholds):
        self.thresholds = thresholds
        self.seen_near = False
        self.crossed = False
        self.first_crossing_s = None

    def update(self, foot_t1_local, timestamp):
        foot = np.asarray(foot_t1_local, float)
        if foot.shape != (3,) or not np.isfinite(foot).all():
            raise ValueError("foot_t1_local must be finite (3,)")
        self.seen_near |= bool(foot[1] > self.thresholds.seen_near_y_m)
        if (
            self.seen_near
            and foot[1] < self.thresholds.crossed_y_m
            and foot[0] < self.thresholds.crossed_x_max_m
        ):
            if not self.crossed:
                self.first_crossing_s = float(timestamp)
            self.crossed = True
        if not self.crossed:
            return "crossing"
        return "post_cross_support_hold"
