"""Closed-loop MuJoCo execution for the Q1/T1 mount-and-ride skill."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from .contracts import load_contract, pd_torque
from .controller import Actor, PermissionActor, PolicyController, ReferencePlayer
from .phases import CrossingThresholds, CrossingTracker


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def qpos_sha256(qpos):
    return hashlib.sha256(np.asarray(qpos, np.float64).tobytes()).hexdigest()


@dataclass
class MountRideConfig:
    root: Path
    raw: dict

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        root = path.parent.parent
        raw = json.loads(path.read_text())
        if raw.get("schema") != "q1-t1-mount-riding-skill-config-v1":
            raise ValueError("unsupported config schema")
        return cls(root, raw)

    def resolve(self, key):
        return (self.root / self.raw["paths"][key]).resolve()

    def verify_assets(self):
        for relative, expected in self.raw["expected_sha256"].items():
            path = self.root / relative
            actual = sha256(path)
            if actual != expected:
                raise ValueError(f"asset hash mismatch: {relative}")


def _contact_metrics(model, data):
    metrics = {
        "left_palm_Fz_N": 0.0,
        "right_palm_Fz_N": 0.0,
        "seat_Fz_N": 0.0,
        "left_foot_Fz_N": 0.0,
        "right_foot_Fz_N": 0.0,
        "unapproved_T1_Fn_N": 0.0,
    }
    for index in range(data.ncon):
        contact = data.contact[index]
        body1 = model.body(int(model.geom_bodyid[contact.geom1])).name
        body2 = model.body(int(model.geom_bodyid[contact.geom2])).name
        pair = {body1, body2}
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, force)
        fn = max(0.0, float(force[0]))
        world_axes = np.asarray(data.contact[index].frame, float).reshape(3, 3)
        # MuJoCo's contact frame first row is the normal on geom1. The world
        # vertical component is obtained from the live contact frame below.
        frame = np.zeros(9, dtype=np.float64)
        mujoco.mj_contactForce(model, data, index, force)
        normal_world = np.asarray(contact.frame[0:3], float)
        # contact.frame stores tangent/normal rows in the world frame.
        vertical_fraction = abs(float(normal_world[2]))
        vertical = fn * vertical_fraction
        if pair == {"q1_pelvis", "BASELINK"}:
            metrics["seat_Fz_N"] += vertical
        elif pair == {"q1_left_wrist_yaw_link", "BASELINK"}:
            metrics["left_palm_Fz_N"] += vertical
        elif pair == {"q1_right_wrist_yaw_link", "BASELINK"}:
            metrics["right_palm_Fz_N"] += vertical
        elif body1 == "world" and body2 == "q1_left_ankle_roll_link":
            metrics["left_foot_Fz_N"] += vertical
        elif body1 == "world" and body2 == "q1_right_ankle_roll_link":
            metrics["right_foot_Fz_N"] += vertical
        q1_bodies = [name for name in (body1, body2) if name.startswith("q1_")]
        t1_contact = any(
            name == "BASELINK"
            or name in {
                "FR_HIP_PITCH_Link",
                "FL_HIP_PITCH_Link",
                "RR_HIP_PITCH_Link",
                "RL_HIP_PITCH_Link",
            }
            for name in (body1, body2)
        )
        approved = pair in (
            {"q1_pelvis", "BASELINK"},
            {"q1_left_wrist_yaw_link", "BASELINK"},
            {"q1_right_wrist_yaw_link", "BASELINK"},
        )
        if q1_bodies and t1_contact and not approved:
            metrics["unapproved_T1_Fn_N"] += fn
    return metrics


class MountRideExecutor:
    def __init__(self, config, dr_profile="none", seed=0):
        self.config = config
        self.raw = config.raw
        self.dr_profile = dr_profile
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        config.verify_assets()
        self.model = mujoco.MjModel.from_xml_path(str(config.resolve("model")))
        if (self.model.nq, self.model.nv, self.model.nu) != (50, 48, 36):
            raise ValueError("unexpected Q1/T1 MuJoCo dimensions")
        if not np.isclose(self.model.opt.timestep, 0.001):
            raise ValueError("native timestep must be 0.001 s")
        self.contract, self.contract_raw = load_contract(config.resolve("contract"))
        self.reference = ReferencePlayer(config.resolve("reference"))
        self.actor = PermissionActor(
            Actor(config.resolve("actor")),
            self.contract.joint_names,
            accepted=int(self.raw["execution"]["candidate_accepted"]),
            mode=self.raw["execution"]["split_left_mode"],
        )
        self.actor.configure(float(self.raw["execution"]["start_reference_s"]))
        self._apply_dr()
        self.data = mujoco.MjData(self.model)
        with np.load(config.resolve("initial_state"), allow_pickle=False) as state:
            self.data.qpos[:] = state["qpos"]
            self.data.qvel[:] = state["qvel"]
            self.history360 = state["history_before_this_policy_tick360"].copy()
            self.previous_action = state[
                "previous_action_before_this_tick"
            ].copy()
        expected_qpos = self.raw["execution"]["initial_qpos_sha256"]
        if qpos_sha256(self.data.qpos) != expected_qpos:
            raise ValueError("initial qpos hash mismatch before perturbation")
        self._perturb_initial_state()
        mujoco.mj_forward(self.model, self.data)
        self.controller = PolicyController(
            self.actor,
            self.reference,
            self.contract,
            self._takeover,
            start_reference_s=float(self.raw["execution"]["start_reference_s"]),
        )
        self.controller.restore_history(self.history360, self.previous_action)
        self.crossing = CrossingTracker(
            CrossingThresholds(**self.raw["crossing"])
        )
        self._prepare_addresses()

    def _apply_dr(self):
        if self.dr_profile == "none":
            return
        dr = json.loads((self.config.root / "configs/dr_light.json").read_text())
        mass_scale = self.rng.uniform(
            dr["mass_scale"]["low"], dr["mass_scale"]["high"]
        )
        friction_scale = self.rng.uniform(
            dr["contact_friction_scale"]["low"],
            dr["contact_friction_scale"]["high"],
        )
        gain_scale = self.rng.uniform(
            dr["pd_gain_scale"]["low"], dr["pd_gain_scale"]["high"]
        )
        for body_id in range(1, self.model.nbody):
            if self.model.body(body_id).name.startswith("q1_"):
                self.model.body_mass[body_id] *= mass_scale
        contact_geoms = np.flatnonzero(
            (self.model.geom_contype != 0)
            | (self.model.geom_conaffinity != 0)
        )
        self.model.geom_friction[contact_geoms, 0] *= friction_scale
        self.contract.kp *= gain_scale
        self.contract.kd *= gain_scale
        self.dr_applied = {
            "mass_scale": float(mass_scale),
            "friction_scale": float(friction_scale),
            "pd_gain_scale": float(gain_scale),
        }

    def _perturb_initial_state(self):
        if self.dr_profile == "none":
            self.initial_perturbation = None
            return
        dr = json.loads((self.config.root / "configs/dr_light.json").read_text())
        max_offset = float(dr["initial_root_offset_m"])
        max_angle = float(dr["initial_joint_angle_rad"])
        offset = self.rng.uniform(-max_offset, max_offset, size=3)
        joint_delta = self.rng.uniform(-max_angle, max_angle, size=22)
        self.data.qpos[21:24] += offset
        joint_qpos = np.asarray(
            [
                self.model.jnt_qposadr[self.model.joint(name).id]
                for name in self.contract.joint_names
            ]
        )
        self.data.qpos[joint_qpos] += joint_delta
        self.initial_perturbation = {
            "root_offset_m": offset.tolist(),
            "joint_delta_rad": joint_delta.tolist(),
        }

    def _prepare_addresses(self):
        self.joint_qpos = np.asarray(
            [
                self.model.jnt_qposadr[self.model.joint(name).id]
                for name in self.contract.joint_names
            ]
        )
        self.joint_qvel = np.asarray(
            [
                self.model.jnt_dofadr[self.model.joint(name).id]
                for name in self.contract.joint_names
            ]
        )
        actuator = {
            int(self.model.actuator_trnid[index, 0]): index
            for index in range(self.model.nu)
        }
        self.actuator_ids = np.asarray(
            [
                actuator[self.model.joint(name).id]
                for name in self.contract.joint_names
            ]
        )
        t1_joints = [
            self.model.joint(name).id
            for name in self.contract_raw["t1_joint_names"]
        ]
        self.t1_qpos = self.model.jnt_qposadr[t1_joints]
        self.t1_qvel = self.model.jnt_dofadr[t1_joints]
        self.t1_actuator = np.asarray(
            [actuator[joint_id] for joint_id in t1_joints]
        )
        self.t1_kp = np.asarray(self.contract_raw["t1_kp"])
        self.t1_kd = np.asarray(self.contract_raw["t1_kd"])
        self.t1_target = np.asarray(self.contract_raw["t1_target"])
        self.pelvis = self.model.body("q1_pelvis").id
        self.t1_base = self.model.body("BASELINK").id
        self.right_foot = self.model.body("q1_right_ankle_roll_link").id

    def _takeover(self, reason):
        raise RuntimeError(f"policy takeover requested: {reason}")

    def _apply_command_inset(self, command):
        command.q_target = command.q_target.astype(np.float32, copy=True)
        ankle_index = self.contract.joint_names.index(
            "q1_right_ankle_pitch_joint"
        )
        ankle_lower = self.model.jnt_range[
            self.model.joint("q1_right_ankle_pitch_joint").id, 0
        ]
        command.q_target[ankle_index] = max(
            command.q_target[ankle_index],
            float(
                ankle_lower
                + self.raw["execution"][
                    "right_ankle_command_lower_inset_rad"
                ]
            ),
        )
        left_roll_index = self.contract.joint_names.index(
            "q1_left_shoulder_roll_joint"
        )
        left_roll_lower = self.model.jnt_range[
            self.model.joint("q1_left_shoulder_roll_joint").id, 0
        ]
        command.q_target[left_roll_index] = max(
            command.q_target[left_roll_index], float(left_roll_lower + 0.020)
        )
        return command

    def run(self, duration_s=None):
        duration = float(duration_s or self.raw["execution"]["duration_s"])
        steps = int(round(duration / self.model.opt.timestep))
        samples = []
        commands = []
        qpos = []
        qvel = []
        times = []
        first_guard = None
        command = None
        for step in range(steps):
            if step % 20 == 0:
                scratch = mujoco.MjData(self.model)
                scratch.qpos[:] = self.data.qpos
                scratch.qvel[:] = self.data.qvel
                mujoco.mj_kinematics(self.model, scratch)
                mujoco.mj_comPos(self.model, scratch)
                gravity = (
                    scratch.xmat[self.pelvis].reshape(3, 3).T
                    @ np.array([0.0, 0.0, -1.0])
                )
                command = self.controller.step(
                    {
                        "timestamp": float(self.data.time),
                        "q": self.data.qpos[self.joint_qpos].copy(),
                        "dq": self.data.qvel[self.joint_qvel].copy(),
                        "body_gyro": self.data.qvel[23:26].copy(),
                        "projected_gravity": gravity,
                    },
                    float(self.data.time),
                )
                if command is None:
                    raise RuntimeError("policy controller requested takeover")
                command = self._apply_command_inset(command)
                commands.append(
                    {
                        "time_s": float(self.data.time),
                        "reference_clock_s": float(
                            self.raw["execution"]["start_reference_s"]
                            + self.data.time
                        ),
                        "normalized_action": command.normalized_action.tolist(),
                        "q_target": command.q_target.tolist(),
                        "permission": self.actor.diagnostic(),
                    }
                )
            torques = pd_torque(
                command,
                self.data.qpos[self.joint_qpos],
                self.data.qvel[self.joint_qvel],
            )
            self.data.ctrl[self.actuator_ids] = torques
            t1_torque = self.t1_kp * (
                self.t1_target - self.data.qpos[self.t1_qpos]
            ) - self.t1_kd * self.data.qvel[self.t1_qvel]
            self.data.ctrl[self.t1_actuator] = np.clip(
                t1_torque,
                self.model.actuator_ctrlrange[self.t1_actuator, 0],
                self.model.actuator_ctrlrange[self.t1_actuator, 1],
            )
            mujoco.mj_step(self.model, self.data)
            times.append(float(self.data.time))
            qpos.append(self.data.qpos.copy())
            qvel.append(self.data.qvel.copy())
            if (step + 1) % 20 == 0:
                rotation = self.data.xmat[self.t1_base].reshape(3, 3)
                foot_local = rotation.T @ (
                    self.data.xpos[self.right_foot]
                    - self.data.xpos[self.t1_base]
                )
                self.actor.state.finish_block(
                    float(
                        self.raw["execution"]["start_reference_s"]
                        + self.data.time
                    ),
                    foot_local,
                    float(self.data.time) - 0.001,
                )
            if step % 20 == 19 or step == steps - 1:
                rotation = self.data.xmat[self.t1_base].reshape(3, 3)
                foot_local = rotation.T @ (
                    self.data.xpos[self.right_foot]
                    - self.data.xpos[self.t1_base]
                )
                phase = self.crossing.update(foot_local, self.data.time)
                metrics = _contact_metrics(self.model, self.data)
                metrics.update(
                    {
                        "time_s": float(self.data.time),
                        "phase": phase,
                        "crossed": self.crossing.crossed,
                        "root_height_m": float(
                            self.data.xpos[self.pelvis, 2]
                        ),
                        "T1_tilt_deg": float(
                            np.degrees(
                                np.arccos(
                                    np.clip(
                                        self.data.xmat[
                                            self.t1_base
                                        ].reshape(3, 3)[2, 2],
                                        -1.0,
                                        1.0,
                                    )
                                )
                            )
                        ),
                        "permission": self.actor.diagnostic(),
                    }
                )
                samples.append(metrics)
            finite = np.isfinite(
                np.r_[self.data.qpos, self.data.qvel, self.data.ctrl]
            ).all()
            root_height = float(self.data.xpos[self.pelvis, 2])
            t1_tilt = float(
                np.degrees(
                    np.arccos(
                        np.clip(
                            self.data.xmat[self.t1_base].reshape(3, 3)[2, 2],
                            -1.0,
                            1.0,
                        )
                    )
                )
            )
            if (
                not finite
                or root_height
                < float(self.raw["safety"]["root_height_min_m"])
                or t1_tilt > float(self.raw["safety"]["t1_tilt_max_deg"])
            ):
                first_guard = {
                    "time_s": float(self.data.time),
                    "finite": bool(finite),
                    "root_height_m": root_height,
                    "T1_tilt_deg": t1_tilt,
                }
                break
        summary = {
            "schema": "q1-t1-closed-loop-run-v1",
            "start_type": self.raw["demo"]["start_type"],
            "ordinary_standing_start_supported": self.raw["demo"][
                "ordinary_standing_start_supported"
            ],
            "duration_s": float(times[-1]),
            "native_ticks": len(times),
            "first_guard": first_guard,
            "crossing_completed": self.crossing.crossed,
            "crossing_complete_s": self.crossing.first_crossing_s,
            "dr_profile": self.dr_profile,
            "seed": self.seed,
            "dr_applied": getattr(self, "dr_applied", None),
            "initial_perturbation": self.initial_perturbation,
            "actor": str(self.config.resolve("actor")),
            "actor_sha256": sha256(self.config.resolve("actor")),
            "model_sha256": sha256(self.config.resolve("model")),
            "physics_steps": len(times),
            "external_forces": False,
            "root_fixed": False,
            "mid_episode_reset": False,
            "formal_legal_hold_claimed": False,
            "commands": commands,
            "samples": samples,
        }
        return {
            "time": np.asarray(times),
            "qpos": np.asarray(qpos),
            "qvel": np.asarray(qvel),
            "summary": summary,
        }

    def run_to_directory(self, output_dir, duration_s=None):
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=False)
        result = self.run(duration_s=duration_s)
        np.savez_compressed(
            output / "trajectory.npz",
            time=result["time"],
            qpos=result["qpos"],
            qvel=result["qvel"],
        )
        (output / "summary.json").write_text(
            json.dumps(result["summary"], indent=2) + "\n"
        )
        return output
