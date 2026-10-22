"""Pure NumPy observation, command, and PD contract."""

from dataclasses import dataclass, fields
import json
from pathlib import Path

import numpy as np


def _vector(value, size, name):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name}: expected finite ({size},), got {result.shape}")
    return result


@dataclass
class ControlContract:
    joint_names: list
    q_nom: object
    action_scale: object
    q_min: object
    q_max: object
    kp: object
    kd: object
    torque_limit: object
    policy_hz: float = 50.0

    def __post_init__(self):
        if len(self.joint_names) != 22 or len(set(self.joint_names)) != 22:
            raise ValueError("22 unique named joints required")
        for name in (
            "q_nom",
            "action_scale",
            "q_min",
            "q_max",
            "kp",
            "kd",
            "torque_limit",
        ):
            setattr(self, name, _vector(getattr(self, name), 22, name))
        if not np.all(self.q_min < self.q_max):
            raise ValueError("invalid joint limits")
        if np.any(self.action_scale <= 0):
            raise ValueError("action scales must be positive")
        if np.any(self.kp < 0) or np.any(self.kd < 0):
            raise ValueError("PD gains must be nonnegative")
        if np.any(self.torque_limit <= 0):
            raise ValueError("torque limits must be positive")
        if self.policy_hz != 50.0:
            raise ValueError("contract v1 requires 50 Hz")


class ObservationBuilder:
    def __init__(self, contract):
        self.contract = contract
        self.history = None

    def reset(self):
        self.history = None

    def build(
        self,
        q,
        dq,
        body_gyro,
        projected_gravity,
        previous_action,
        reference_q_3,
        reference_gravity,
        phase,
    ):
        contract = self.contract
        previous = _vector(previous_action, 22, "previous_action")
        if np.any(np.abs(previous) > 1.000001):
            raise ValueError("previous action must be clipped normalized output")
        frame = np.concatenate(
            (
                _vector(q, 22, "q") - contract.q_nom,
                _vector(dq, 22, "dq"),
                _vector(body_gyro, 3, "body_gyro"),
                _vector(projected_gravity, 3, "projected_gravity"),
                previous,
            )
        )
        if self.history is None:
            self.history = np.repeat(frame[None], 5, axis=0)
        else:
            self.history = np.concatenate((self.history[1:], frame[None]))
        reference = np.asarray(reference_q_3, dtype=np.float32)
        if reference.shape != (3, 22) or not np.isfinite(reference).all():
            raise ValueError("reference_q_3 must be finite (3,22)")
        if not np.isfinite(phase) or not 0.0 <= phase <= 1.0:
            raise ValueError("phase outside [0,1]")
        observation = np.concatenate(
            (
                self.history.ravel(),
                reference.ravel(),
                _vector(reference_gravity, 3, "reference_gravity"),
                np.array([phase], dtype=np.float32),
            )
        )
        if observation.shape != (430,):
            raise ValueError("observation contract must produce 430 values")
        return observation.astype(np.float32, copy=False)


@dataclass
class JointCommand:
    q_target: object
    dq_target: object
    kp: object
    kd: object
    torque_limit: object
    normalized_action: object
    status: str


def map_action(action, reference_q, contract, status="running"):
    normalized = np.clip(_vector(action, 22, "action"), -1.0, 1.0)
    target = np.clip(
        _vector(reference_q, 22, "reference_q")
        + contract.action_scale * normalized,
        contract.q_min,
        contract.q_max,
    )
    return JointCommand(
        target,
        np.zeros(22, np.float32),
        contract.kp.copy(),
        contract.kd.copy(),
        contract.torque_limit.copy(),
        normalized,
        status,
    )


def pd_torque(command, q, dq):
    raw = command.kp * (
        command.q_target - _vector(q, 22, "q")
    ) + command.kd * (command.dq_target - _vector(dq, 22, "dq"))
    return np.clip(raw, -command.torque_limit, command.torque_limit)


def load_contract(path):
    raw = json.loads(Path(path).read_text())
    if raw.get("obs_dim") != 430:
        raise ValueError("unsupported observation dimension")
    if raw.get("normalization") != "identity":
        raise ValueError("unsupported observation normalizer")
    if raw.get("native_dt") != 0.001:
        raise ValueError("unsupported native timestep")
    names = {field.name for field in fields(ControlContract)}
    return ControlContract(
        **{key: raw[key] for key in names if key in raw}
    ), raw
