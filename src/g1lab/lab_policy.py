"""CPU MuJoCo adapter for the full-body Unitree RL Lab ONNX velocity policy.

Reimplements its deployment contract; not the 12-leg-DoF RL Gym teacher.
Vendor weights and assets are fetched separately and are never redistributed here.
"""

import ast
from pathlib import Path

import mujoco
import numpy as np


class TermHistory:
    """Term-major history, oldest to newest, as in Unitree's C++ observation manager."""

    def __init__(self, configuration):
        self.configuration = configuration
        self.buffers = {}

    def _value(self, name, value):
        cfg = self.configuration[name]
        value = np.asarray(value, dtype=np.float32)
        if cfg.get("clip") is not None:
            value = np.clip(value, *cfg["clip"])
        return value * np.asarray(cfg.get("scale", 1), dtype=np.float32)

    def reset(self, values):
        self.buffers = {
            name: np.repeat(self._value(name, values[name])[None], cfg.get("history_length", 1), axis=0)
            for name, cfg in self.configuration.items()
        }

    def update(self, values):
        for name, buffer in self.buffers.items():
            buffer[:-1] = buffer[1:]
            buffer[-1] = self._value(name, values[name])
        return np.concatenate([v.ravel() for v in self.buffers.values()]).astype(np.float32)


def sdk_names(source):
    path = Path(source) / "source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py"
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "UNITREE_G1_29DOF_CFG" for t in node.targets
        ):
            return ast.literal_eval(next(k.value for k in node.value.keywords if k.arg == "joint_sdk_names"))
    raise ValueError("Cannot find upstream SDK joint names")


class LabReference:
    def __init__(
        self, source="vendor/unitree_rl_lab", mujoco_source="vendor/unitree_mujoco", policy_dir=None
    ):
        import onnxruntime as ort
        import yaml

        self.source = Path(source)
        folder = (
            Path(policy_dir)
            if policy_dir
            else self.source / "deploy/robots/g1_29dof/config/policy/velocity/v0"
        )
        self.policy_path = folder / "exported/policy.onnx"
        self.config = yaml.safe_load((folder / "params/deploy.yaml").read_text())
        self.model = mujoco.MjModel.from_xml_path(
            str(Path(mujoco_source) / "unitree_robots/g1/scene_29dof.xml")
        )
        self.model.opt.timestep = 0.002
        self.data = mujoco.MjData(self.model)
        self.pelvis = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        mapping = np.array(self.config["joint_ids_map"])
        names = sdk_names(source)
        if len(names) != 29 or sorted(mapping.tolist()) != list(range(29)):
            raise ValueError("Invalid upstream joint permutation")
        ordered_names = [names[i] for i in mapping]
        joint_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ordered_names]
        )
        if (joint_ids < 0).any():
            raise ValueError("MuJoCo model does not match upstream joint names")
        actuator_for_joint = {int(j): i for i, j in enumerate(self.model.actuator_trnid[:, 0])}
        self.actuator_ids = np.array([actuator_for_joint[int(j)] for j in joint_ids])
        self.foot_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll_link")
            for side in ("left", "right")
        ]
        if min(self.foot_ids) < 0:
            raise ValueError("Missing foot bodies in MuJoCo model")
        self.joint_index = {name: i for i, name in enumerate(ordered_names)}
        self.knee_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_knee_link")
            for s in ("left", "right")
        ]
        self.torso = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.body_weight = float(self.model.body_mass.sum() * 9.81)
        self.gait_log = []
        self.qadr = self.model.jnt_qposadr[joint_ids]
        self.vadr = self.model.jnt_dofadr[joint_ids]
        self.kp = np.array(self.config["stiffness"])[mapping]
        self.kd = np.array(self.config["damping"])[mapping]
        self.nominal = np.array(self.config["default_joint_pos"])
        self.scale = np.array(self.config["actions"]["JointPositionAction"]["scale"])
        self.offset = np.array(self.config["actions"]["JointPositionAction"]["offset"])
        self.control_dt = float(self.config["step_dt"])
        if self.control_dt != 0.02:
            raise ValueError("Unexpected policy control period")
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self.policy_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        if self.session.get_inputs()[0].shape != [1, 480] or self.session.get_outputs()[0].shape != [1, 29]:
            raise ValueError("Unexpected ONNX observation/action shape")
        self.history = TermHistory(self.config["observations"])
        self.action = np.zeros(29)
        self.command = np.zeros(3)

    def values(self):
        rotation = self.data.xmat[self.pelvis].reshape(3, 3)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.pelvis, velocity, 1)
        return {
            "base_ang_vel": velocity[:3],
            "projected_gravity": rotation.T @ [0, 0, -1],
            "velocity_commands": self.command,
            "joint_pos_rel": self.data.qpos[self.qadr] - self.nominal,
            "joint_vel_rel": self.data.qvel[self.vadr],
            "last_action": self.action,
        }

    def reset(self, seed=42):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.8
        self.data.qpos[self.qadr] = self.nominal + np.random.default_rng(seed).uniform(-0.005, 0.005, 29)
        self.data.qvel[:] = 0
        self.action[:] = 0
        self.command[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.history.reset(self.values())
        self.gait_log = []

    def _gait_sample(self):
        """(time, contact L, R, toe L, R, heel L, R, forward ground force N, normal force L, R); read-only.

        A foot is in contact when its summed normal force exceeds 20 N (same threshold as the Isaac sprint
        sensor); toe/heel flags mark which contact spheres carry load (> 1 N).
        """
        normal, toe, heel = [0.0, 0.0], [False, False], [False, False]
        rotation = self.data.xmat[self.pelvis].reshape(3, 3)
        heading = np.array([rotation[0, 0], rotation[1, 0], 0.0])
        heading /= max(np.linalg.norm(heading), 1e-9)
        forward = 0.0
        wrench = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1, b2 = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
            if 0 not in (b1, b2):
                continue
            foot_body, foot_geom, sign = (b2, c.geom2, 1.0) if b1 == 0 else (b1, c.geom1, -1.0)
            if foot_body not in self.foot_ids:
                continue
            mujoco.mj_contactForce(self.model, self.data, i, wrench)
            if wrench[0] <= 1.0:
                continue
            f = self.foot_ids.index(foot_body)
            normal[f] += float(wrench[0])
            if self.model.geom_pos[foot_geom][0] > 0.03:
                toe[f] = True
            else:
                heel[f] = True
            world = c.frame.reshape(3, 3).T @ wrench[:3]
            forward += sign * float(world @ heading)
        contact = [n > 20.0 for n in normal]
        return (float(self.data.time), *contact, *toe, *heel, forward, *normal)

    def step(self, command, diagnostics=False):
        self.command[:] = command
        observation = self.history.update(self.values())
        self.action = self.session.run(None, {self.session.get_inputs()[0].name: observation[None]})[0][0]
        if not np.isfinite(self.action).all():
            raise FloatingPointError("Non-finite ONNX action")
        target = self.offset + self.action * self.scale
        clipped = 0
        peak_utilization = 0.0
        limits = self.model.actuator_ctrlrange[self.actuator_ids]
        for _ in range(10):
            self.data.ctrl[self.actuator_ids] = (
                self.kp * (target - self.data.qpos[self.qadr]) - self.kd * self.data.qvel[self.vadr]
            )
            if diagnostics:
                requested = self.data.ctrl[self.actuator_ids]
                clipped += np.count_nonzero((requested < limits[:, 0]) | (requested > limits[:, 1]))
            mujoco.mj_step(self.model, self.data)
            if diagnostics:
                self.gait_log.append(self._gait_sample())
                peak_utilization = max(
                    peak_utilization,
                    float(
                        np.max(
                            np.abs(self.data.actuator_force[self.actuator_ids])
                            / np.max(np.abs(limits), axis=1)
                        )
                    ),
                )
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.pelvis, velocity, 1)
        upright = float(self.data.xmat[self.pelvis].reshape(3, 3)[2, 2])
        finite = np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
        result = {
            "time_s": float(self.data.time),
            "vx_mps": float(velocity[3]),
            "vy_mps": float(velocity[4]),
            "yaw_rate": float(velocity[2]),
            "height_m": float(self.data.qpos[2]),
            "upright_cos": upright,
            "fallen": bool(not finite or self.data.qpos[2] < 0.45 or upright < 0.45),
            "command_vx_mps": float(command[0]),
        }
        if diagnostics:
            touching = set()
            for contact in self.data.contact:
                if contact.dist <= 0:
                    bodies = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
                    if 0 in bodies:
                        touching.update(int(b) for b in bodies if b in self.foot_ids)
            speeds = []
            for foot in touching:
                mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, foot, velocity, 0)
                speeds.append(np.linalg.norm(velocity[3:5]))
            last = self.gait_log[-1]
            torso = self.data.xmat[self.torso].reshape(3, 3)
            pelvis_z = self.data.xpos[self.pelvis][2]
            for i, side in enumerate(("l", "r")):
                name = "left" if side == "l" else "right"
                result[f"contact_{side}"] = bool(last[1 + i])
                result[f"knee_drop_{side}_m"] = float(pelvis_z - self.data.xpos[self.knee_ids[i]][2])
                k = self.joint_index[f"{name}_ankle_pitch_joint"]
                result[f"ankle_pitch_rel_{side}_rad"] = float(self.data.qpos[self.qadr[k]] - self.nominal[k])
                for joint in ("shoulder_pitch", "hip_pitch"):
                    j = self.joint_index[f"{name}_{joint}_joint"]
                    result[f"{joint}_{side}_rad"] = float(self.data.qpos[self.qadr[j]])
            result["torso_pitch_deg"] = float(
                np.degrees(np.arcsin(np.clip((torso.T @ [0, 0, -1])[0], -1, 1)))
            )
            result.update(
                {
                    "requested_torque_clipped_fraction": clipped / (10 * 29),
                    "peak_torque_utilization": peak_utilization,
                    "feet_in_contact": len(touching),
                    "contact_foot_speed_mps": float(np.mean(speeds)) if speeds else 0.0,
                }
            )
        return result
