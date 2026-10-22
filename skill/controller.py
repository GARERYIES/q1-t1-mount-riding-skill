"""Deterministic policy controller and frozen arm-authority state."""

from pathlib import Path

import numpy as np
import torch

from .authority import split_left_authority as split
from .authority import postcross_authority as right_authority
from .contracts import ObservationBuilder, map_action


class Actor:
    def __init__(self, path):
        path = Path(path)
        if path.suffix not in (".ts", ".jit"):
            raise ValueError("verified TorchScript actor required")
        self.model = torch.jit.load(str(path), map_location="cpu").eval()

    def warmup(self, repeats=5):
        if not 3 <= repeats <= 20:
            raise ValueError("warmup requires 3..20 bounded repeats")
        timings = []
        for _ in range(repeats):
            self(np.zeros(430, np.float32))
        return {"ready": True, "repeats": repeats, "commands_sent": 0}

    def __call__(self, observation):
        value = np.asarray(observation, np.float32)
        if value.shape != (430,) or not np.isfinite(value).all():
            raise ValueError("actor requires finite observation with shape (430,)")
        with torch.inference_mode():
            result = self.model(torch.from_numpy(value[None])).cpu().numpy()
        result = np.asarray(result, np.float32)
        if result.shape != (1, 22) or not np.isfinite(result).all():
            raise ValueError("invalid actor output")
        return result[0]


class ReferencePlayer:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as source:
            self.time = np.asarray(source["time"], float)
            self.q = np.asarray(source["q1_joint_positions"], np.float32)
            self.qpos = np.asarray(source["qpos"], float)
            self.names = [str(value) for value in source["joint_names"]]
        if self.qpos.shape != (len(self.time), 50):
            raise ValueError("invalid reference qpos")
        if self.q.shape != (len(self.time), 22):
            raise ValueError("invalid reference joint positions")
        if len(self.time) < 2 or not np.all(np.diff(self.time) > 0):
            raise ValueError("invalid reference clock")
        if not all(
            np.isfinite(value).all()
            for value in (self.time, self.q, self.qpos)
        ):
            raise ValueError("nonfinite reference")
        if np.any(np.linalg.norm(self.qpos[:, 24:28], axis=1) < 1e-8):
            raise ValueError("zero reference quaternion")
        self.qpos = self.qpos.copy()
        for index in range(1, len(self.time)):
            if (
                np.dot(
                    self.qpos[index - 1, 24:28],
                    self.qpos[index, 24:28],
                )
                < 0
            ):
                self.qpos[index, 24:28] *= -1
        self.time = self.time - self.time[0]
        self.duration = float(self.time[-1])

    def sample(self, timestamp):
        return np.asarray(
            [
                np.interp(timestamp, self.time, self.q[:, joint])
                for joint in range(22)
            ],
            np.float32,
        )

    def gravity(self, timestamp):
        quat = np.asarray(
            [
                np.interp(timestamp, self.time, self.qpos[:, index])
                for index in range(24, 28)
            ]
        )
        quat /= np.linalg.norm(quat)
        w, x, y, z = quat
        return np.asarray(
            [2 * (w * y - x * z), -2 * (y * z + w * x), -(1 - 2 * (x * x + y * y))],
            np.float32,
        )


class SplitState:
    def __init__(self, phase, accepted=16, mode="B"):
        if accepted != 16 or mode not in ("A", "B"):
            raise ValueError("the shipped authority requires accepted=16 and mode A/B")
        self.phase = float(phase)
        self.crossed = False
        self.seen_near = False
        self.accepted = 16
        self.role = True
        self.mode = mode
        self.ticks = 0
        self.was_eligible = False
        self.last_measurement = None
        self.left_ticks = 0
        self.left_mapping_ticks = 0
        self.left_was_eligible = False
        self.left_mapping_was_eligible = False

    def gains8(self):
        return split.split_gains(
            torch.tensor([self.role]),
            torch.tensor([self.crossed]),
            torch.tensor([self.phase], dtype=torch.float64),
            torch.tensor([self.ticks]),
            torch.tensor([self.left_ticks]),
            self.mode,
        )[0]

    def finish_block(self, phase, foot_t1_local, geometry_time_s):
        self.phase = float(phase)
        foot = np.asarray(foot_t1_local, float)
        self.seen_near |= bool(foot[1] > 0.05)
        # Mirrors the frozen source crossing detector used by the validated run.
        self.crossed |= bool(
            self.seen_near and foot[1] < -0.05 and foot[0] < -0.18
        )
        ticks, eligible = right_authority.advance_permission_ticks(
            torch.tensor([self.ticks]),
            torch.tensor([self.was_eligible]),
            torch.tensor([self.role]),
            torch.tensor([self.crossed]),
            torch.tensor([self.phase], dtype=torch.float64),
        )
        self.ticks = int(ticks[0])
        self.was_eligible = bool(eligible[0])
        eligible = split.left_eligible(
            torch.tensor([self.role]),
            torch.tensor([self.phase], dtype=torch.float64),
            self.mode,
            torch.tensor([self.crossed]),
        )
        ticks, eligible = split.advance_left_ticks(
            torch.tensor([self.left_ticks]),
            torch.tensor([self.left_was_eligible]),
            eligible,
        )
        self.left_ticks = int(ticks[0])
        self.left_was_eligible = bool(eligible[0])
        ticks, eligible = split.advance_left_ticks(
            torch.tensor([self.left_mapping_ticks]),
            torch.tensor([self.left_mapping_was_eligible]),
            eligible,
        )
        self.left_mapping_ticks = int(ticks[0])
        self.left_mapping_was_eligible = bool(eligible[0])
        self.last_measurement = {
            "foot_T1_local_m": foot.tolist(),
            "force_geometry_time_s": float(geometry_time_s),
        }

    def diagnostic(self):
        gains = self.gains8().tolist()
        return {
            "role": self.role,
            "mode": self.mode,
            "crossed": self.crossed,
            "seen_near": self.seen_near,
            "permission_age_ticks": self.ticks,
            "was_eligible": self.was_eligible,
            "gains8": gains,
            "left_permission_age_ticks": self.left_ticks,
            "left_mapping_age_ticks": self.left_mapping_ticks,
            "phase_s": self.phase,
            "last_native_pre_geometry": self.last_measurement,
            "no_oracle_actor_observations": True,
        }


class PermissionActor:
    def __init__(self, actor, joint_names, accepted=16, mode="B"):
        self.actor = actor
        self.names = joint_names
        self.accepted = accepted
        self.mode = mode
        self.state = None
        self.reference = None
        self.contract = None
        self.last_actor_unmapped = None
        self.last_actor_output = None
        self.last_mapping_diagnostic = None
        self.last_target_before = None
        self.indices = np.arange(14, 22)

    def configure(self, phase):
        self.state = SplitState(phase, self.accepted, self.mode)

    def __call__(self, observation):
        if self.reference is None or self.contract is None:
            raise RuntimeError("controller must attach reference and contract")
        raw = np.asarray(self.actor(observation), np.float32)
        self.last_actor_unmapped = raw.copy()
        contract = self.contract
        gains = self.state.gains8()
        applied, before, _, diagnostics = split.split_target(
            torch.from_numpy(raw.copy())[None],
            torch.from_numpy(self.reference.sample(self.state.phase))[None],
            torch.from_numpy(contract.action_scale),
            torch.from_numpy(contract.q_min),
            torch.from_numpy(contract.q_max),
            float(contract.q_min[10]) - 0.001,
            torch.tensor([self.state.role]),
            gains[None],
            torch.tensor([self.state.left_mapping_ticks]),
        )
        self.last_mapping_diagnostic = {
            key: float(value[0]) for key, value in diagnostics.items()
        }
        self.last_actor_output = np.clip(raw, -1, 1).copy()
        self.last_actor_output[15] = self.last_mapping_diagnostic[
            "left_roll_mapped_action"
        ]
        self.last_target_before = before[0].numpy().copy()
        self.indices = np.flatnonzero(gains.numpy() == 0) + 14
        return applied[0].numpy().copy()

    def warmup(self):
        return self.actor.warmup()

    def diagnostic(self):
        return self.state.diagnostic()


class PolicyController:
    def __init__(self, actor, reference, contract, takeover, start_reference_s=0.0):
        if reference.names != contract.joint_names:
            raise ValueError("reference and control-contract joint order mismatch")
        self.actor = actor
        self.reference = reference
        self.contract = contract
        self.takeover = takeover
        self.builder = ObservationBuilder(contract)
        self.previous_action = np.zeros(22, np.float32)
        self.start = -float(start_reference_s)
        self.last = -0.02
        self.status = "ready"
        self.last_observation = None
        actor.reference = reference
        actor.contract = contract

    def restore_history(self, history360, previous_action):
        history = np.asarray(history360, np.float32)
        if history.shape != (360,):
            raise ValueError("history360 must have shape (360,)")
        self.builder.history = history.reshape(5, 72).copy()
        self.previous_action = np.asarray(previous_action, np.float32).copy()
        if self.previous_action.shape != (22,):
            raise ValueError("previous action must have shape (22,)")

    def step(self, robot_state, timestamp):
        try:
            if not np.isfinite(timestamp):
                raise ValueError("invalid timestamp")
            if self.last is not None and not 0 < timestamp - self.last <= 0.04:
                raise ValueError("control timestamp gap")
            if abs(timestamp - float(robot_state["timestamp"])) > 0.04:
                raise ValueError("stale robot state")
            elapsed = timestamp - self.start
            self.last = timestamp
            self.actor.state.phase = float(elapsed)
            reference = self.reference
            phase = min(1.0, max(0.0, elapsed / reference.duration))
            reference_q = np.stack(
                [reference.sample(elapsed + offset) for offset in (0, 0.1, 0.2)]
            )
            observation = self.builder.build(
                robot_state["q"],
                robot_state["dq"],
                robot_state["body_gyro"],
                robot_state["projected_gravity"],
                self.previous_action,
                reference_q,
                reference.gravity(elapsed),
                phase,
            )
            self.last_observation = observation.copy()
            action = np.asarray(self.actor(observation), np.float32)
            command = map_action(action, reference_q[0], self.contract)
            # The validated adapter publishes the arm target before the final
            # ankle inset is applied by the execution loop.
            command.q_target = self.actor.last_target_before.copy()
            self.previous_action = command.normalized_action.copy()
            return command
        except (ValueError, KeyError, FloatingPointError, RuntimeError, TypeError) as error:
            self.status = "takeover"
            self.takeover(str(error))
            return None
