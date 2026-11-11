"""CPU MuJoCo G1 environment: 29 joint targets, body-frame commands, terrain scan."""

import xml.etree.ElementTree as ET
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .config import EnvConfig
from .terrain import add_terrain

ASSET_DIR = Path(__file__).resolve().parent / "assets" / "g1"


class G1Env(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}

    def __init__(self, config=None, render_mode=None):
        self.config = config or EnvConfig()
        self.render_mode = render_mode
        self.metadata = dict(self.metadata, render_fps=round(1 / self.config.control_dt))
        root = ET.parse(ASSET_DIR / "g1.xml").getroot()
        root.find(".//default[@class='collision']/geom").set("group", "1")
        compiler = root.find("compiler")
        compiler.set("meshdir", str(ASSET_DIR / "assets"))
        option = root.find("option")
        if option is None:
            option = ET.SubElement(root, "option")
        option.set("timestep", str(self.config.physics_dt))
        ET.SubElement(root.find("worldbody"), "light", pos="0 0 5", dir="0 0 -1", directional="true")
        self.has_hfield = add_terrain(root, self.config.terrain, self.config.difficulty)
        # Group 3 isolates terrain for ray casting, excluding robot collision/visual meshes.
        for geom in root.find("worldbody").findall("geom"):
            geom.set("group", "3")
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = round(self.config.control_dt / self.config.physics_dt)
        self.joint_ids = self.model.actuator_trnid[:, 0].copy()
        self.qadr = self.model.jnt_qposadr[self.joint_ids]
        self.vadr = self.model.jnt_dofadr[self.joint_ids]
        self.home = self.model.key_qpos[0].copy()
        if self.config.crouched_home:
            self.home[self.qadr[:12]] = [-0.1, 0, 0, 0.3, -0.2, 0] * 2
        self.nominal = self.home[self.qadr].copy()
        if self.config.unitree_pd:
            gains = np.array([100, 100, 100, 150, 40, 40] * 2)
            damping = np.array([2, 2, 2, 4, 2, 2] * 2)
            self.model.actuator_gainprm[:12, 0] = gains
            self.model.actuator_biasprm[:12, 1] = -gains
            self.model.actuator_biasprm[:12, 2] = -damping
            self.model.dof_damping[self.vadr] = 0.001
            self.model.dof_frictionloss[self.vadr] = 0.1
        self.limits = self.model.jnt_range[self.joint_ids].copy()
        self.base_mass = self.model.body_mass.copy()
        self.base_inertia = self.model.body_inertia.copy()
        self.base_friction = self.model.geom_friction.copy()
        self.base_gain = self.model.actuator_gainprm.copy()
        self.base_bias = self.model.actuator_biasprm.copy()
        self.pelvis = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.foot_sites = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"{s}_foot") for s in ("left", "right")
        ]
        self.ground_geoms = np.flatnonzero(self.model.geom_group == 3)
        self.terrain_group = np.array([0, 0, 0, 1, 0, 0], dtype=np.uint8)
        self.scan_xy = np.array([(x, y) for x in (-0.3, 0.0, 0.3, 0.6, 0.9) for y in (-0.3, 0.0, 0.3)])
        self.action_space = spaces.Box(-1.0, 1.0, (self.model.nu,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            (
                3
                + 3
                + 3
                + 3 * self.model.nu
                + 15
                + (3 if self.config.linear_velocity_observation else 0)
                + (2 if self.config.phase_observation else 0),
            ),
            dtype=np.float32,
        )
        self.previous_action = np.zeros(self.model.nu)
        self.command = np.array(self.config.command, dtype=float)
        self.renderer = self.viewer = None
        self.scene_option = mujoco.MjvOption()
        self.scene_option.geomgroup[1] = 0
        self.scene_option.geomgroup[3] = 1
        self.manual_impulse = np.zeros(2)
        self.step_count = 0

    def ground_height(self, x, y):
        if self.config.terrain == "flat":
            return 0.0
        if self.config.terrain == "slope":
            return float(x * np.tan(np.deg2rad(10 * self.config.difficulty)))
        distance = mujoco.mj_ray(
            self.model,
            self.data,
            np.array([x, y, 20.0]),
            np.array([0.0, 0.0, -1.0]),
            self.terrain_group,
            1,
            -1,
            np.array([-1], dtype=np.int32),
        )
        return 0.0 if distance < 0 else 20.0 - distance

    def body_velocity(self):
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.pelvis, velocity, 1)
        return velocity[3:], velocity[:3]

    def _observation(self):
        rotation = self.data.xmat[self.pelvis].reshape(3, 3)
        gravity = rotation.T @ np.array([0.0, 0.0, -1.0])
        linear, angular = self.body_velocity()
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        scan_xy = self.scan_xy @ np.array([[c, s], [-s, c]])
        position = self.data.qpos[:3]
        scan = [
            np.clip(position[2] - self.ground_height(*(position[:2] + xy)) - 0.75, -1, 1) for xy in scan_xy
        ]
        obs = np.concatenate(
            [
                angular * 0.25,
                gravity,
                self.command,
                self.data.qpos[self.qadr] - self.nominal,
                self.data.qvel[self.vadr] * 0.05,
                self.previous_action,
                scan,
            ]
        )
        if self.config.linear_velocity_observation:
            obs = np.concatenate([obs, linear])
        if self.config.phase_observation:
            phase = 2 * np.pi * (self.data.time % 0.8) / 0.8
            obs = np.concatenate([obs, [np.sin(phase), np.cos(phase)]])
        return obs.astype(np.float32)

    def set_command(self, vx, vy, yaw_rate):
        self.command[:] = (vx, vy, yaw_rate)

    def push(self, impulse=5.0, angle=0.0):
        """Apply a horizontal impulse in world coordinates over the next control step."""
        self.manual_impulse += impulse * np.array([np.cos(angle), np.sin(angle)])

    def _sample_command(self):
        self.command = np.array(
            [
                self.np_random.uniform(-0.2, 0.8),
                self.np_random.uniform(-0.25, 0.25),
                self.np_random.uniform(-0.6, 0.6),
            ]
        )
        if self.np_random.random() < 0.2:
            self.command[:] = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        cfg = self.config
        mujoco.mj_resetData(self.model, self.data)
        mass_scale = cfg.mass_scale
        friction = cfg.friction
        strength = 1.0
        if cfg.domain_randomization:
            mass_scale *= self.np_random.uniform(0.9, 1.1)
            friction *= self.np_random.uniform(0.6, 1.4)
            strength = self.np_random.uniform(0.9, 1.1)
        self.model.body_mass[:] = self.base_mass * mass_scale
        self.model.body_inertia[:] = self.base_inertia * mass_scale
        self.model.geom_friction[:] = self.base_friction
        self.model.geom_friction[:, 0] = friction
        self.model.actuator_gainprm[:] = self.base_gain * strength
        self.model.actuator_biasprm[:] = self.base_bias * strength
        if self.has_hfield:
            if self.renderer is not None:
                self.renderer.close()
                self.renderer = None
            # Smooth seeded noise. A flat central spawn patch prevents initial penetration.
            coarse = self.np_random.uniform(0, 1, (17, 17))
            axis = np.linspace(0, 16, 129)
            rows = np.array([np.interp(axis, np.arange(17), row) for row in coarse])
            field = np.array([np.interp(axis, np.arange(17), col) for col in rows.T]).T
            field[61:68, 61:68] = 0
            self.model.hfield_data[:] = field.ravel()
            if self.viewer is not None:
                self.viewer.update_hfield(0)
        mujoco.mj_setConst(self.model, self.data)
        self.data.qpos[:] = self.home
        self.data.qpos[self.qadr] += self.np_random.uniform(-0.015, 0.015, self.model.nu)
        self.data.ctrl[:] = self.nominal
        mujoco.mj_forward(self.model, self.data)
        # Raise the lowest foot to terrain clearance before starting physics.
        clearance = min(
            self.data.site_xpos[site, 2] - self.ground_height(*self.data.site_xpos[site, :2])
            for site in self.foot_sites
        )
        self.data.qpos[2] += max(0, 0.04 - clearance)
        mujoco.mj_forward(self.model, self.data)
        self.previous_action[:] = 0
        self.manual_impulse[:] = 0
        self.step_count = 0
        self.command = np.array(cfg.command, dtype=float)
        if cfg.random_commands:
            self._sample_command()
        self.next_push = self.np_random.uniform(3, 6)
        self.spawn_x = float(self.data.qpos[0])
        if cfg.task == "recovery":
            from .recovery import reset_recovery

            reset_recovery(self)
        return self._observation(), {}

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError("Action must be finite and match action_space")
        action = np.clip(action, -1, 1)
        if self.config.leg_only:
            action[12:] = 0
        cfg = self.config
        self.data.ctrl[:] = np.clip(
            self.nominal + action * cfg.action_scale, self.limits[:, 0], self.limits[:, 1]
        )
        if cfg.pushes and self.data.time >= self.next_push:
            self.push(self.np_random.uniform(2, 8), self.np_random.uniform(-np.pi, np.pi))
            self.next_push = self.data.time + self.np_random.uniform(3, 6)
        start_xy = self.data.qpos[:2].copy()
        energy = 0.0
        applied_impulse = 0.0
        for _ in range(self.frame_skip):
            self.data.xfrc_applied[:] = 0
            force = self.manual_impulse / cfg.control_dt
            overlap = max(
                0,
                min(self.data.time + cfg.physics_dt, cfg.push_at + cfg.push_duration)
                - max(self.data.time, cfg.push_at),
            )
            if overlap:
                force = force + (cfg.push_impulse / cfg.push_duration) * (
                    overlap / cfg.physics_dt
                ) * np.array([np.cos(cfg.push_angle), np.sin(cfg.push_angle)])
            self.data.xfrc_applied[self.pelvis, :2] = force
            applied_impulse += float(np.linalg.norm(force) * cfg.physics_dt)
            mujoco.mj_step(self.model, self.data)
            energy += (
                float(np.sum(np.abs(self.data.qfrc_actuator[self.vadr] * self.data.qvel[self.vadr])))
                * cfg.physics_dt
            )
        self.manual_impulse[:] = 0
        self.data.xfrc_applied[:] = 0
        self.step_count += 1
        contacting_feet = set()
        for contact in self.data.contact:
            first, second = int(contact.geom1), int(contact.geom2)
            for foot in self.foot_sites:
                body = self.model.site_bodyid[foot]
                if (first in self.ground_geoms and self.model.geom_bodyid[second] == body) or (
                    second in self.ground_geoms and self.model.geom_bodyid[first] == body
                ):
                    contacting_feet.add(foot)
        slip_sum = 0.0
        for foot in contacting_feet:
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, foot, velocity, 0)
            slip_sum += float(np.linalg.norm(velocity[3:5]))
        linear, angular = self.body_velocity()
        error = linear[:2] - self.command[:2]
        yaw_error = angular[2] - self.command[2]
        upright = self.data.xmat[self.pelvis].reshape(3, 3)[2, 2]
        height = self.data.qpos[2] - self.ground_height(*self.data.qpos[:2])
        finite = np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
        fallen = height < 0.45 or upright < 0.45 or not finite
        reward_terms = {
            "tracking": 2.0 * np.exp(-np.dot(error, error) / cfg.tracking_sigma),
            "yaw_tracking": 0.5 * np.exp(-(yaw_error**2) / 0.25),
            "upright": 0.5 * upright,
            "alive": 0.5,
            "action_rate": -0.05 * float(np.mean((action - self.previous_action) ** 2)),
            "power": -0.0005 * energy / cfg.control_dt,
            "vertical_velocity": -0.2 * linear[2] ** 2,
            "fall": -5.0 if fallen else 0.0,
        }
        self.previous_action[:] = action
        info = {
            "velocity_error_sq": float(np.dot(error, error)),
            "yaw_error_sq": float(yaw_error**2),
            "vx": float(linear[0]),
            "vy": float(linear[1]),
            "yaw_rate": float(angular[2]),
            "command_vx": float(self.command[0]),
            "command_vy": float(self.command[1]),
            "command_yaw": float(self.command[2]),
            "energy_j": energy,
            "distance_m": float(np.linalg.norm(self.data.qpos[:2] - start_xy)),
            "x_progress_m": float(self.data.qpos[0] - self.spawn_x),
            "upright": float(upright),
            "base_height_m": float(height),
            "fallen": bool(fallen),
            "impulse_ns": applied_impulse,
            "time": float(self.data.time),
        }
        info.update({f"reward_{name}": float(value) for name, value in reward_terms.items()})
        info.update(foot_slip_sum_mps=slip_sum, contacting_feet=len(contacting_feet))
        for side, site in zip(("left", "right"), self.foot_sites):
            info[f"{side}_foot_contact"] = site in contacting_feet
            info[f"{side}_foot_clearance_m"] = float(
                self.data.site_xpos[site, 2] - self.ground_height(*self.data.site_xpos[site, :2])
            )
        truncated = self.step_count >= round(cfg.episode_seconds / cfg.control_dt)
        if (
            cfg.random_commands
            and self.step_count % max(1, round(cfg.command_interval / cfg.control_dt)) == 0
        ):
            self._sample_command()
        obs = self._observation()
        if not np.isfinite(obs).all():
            raise FloatingPointError("MuJoCo produced non-finite observation")
        if self.render_mode == "human":
            self.render()
        if cfg.task == "recovery":
            from .recovery import recovery_reward

            reward, success = recovery_reward(self, action, info)
            return obs, reward, success, truncated, info
        return obs, float(sum(reward_terms.values())), bool(fallen), truncated, info

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                from mujoco import viewer

                self.viewer = viewer.launch_passive(self.model, self.data)
                self.viewer.opt.geomgroup[1] = 0
                self.viewer.opt.geomgroup[3] = 1
            self.viewer.sync()
            return None
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = self.data.qpos[:3]
        camera.distance, camera.azimuth, camera.elevation = 3.0, 135, -15
        self.renderer.update_scene(self.data, camera, scene_option=self.scene_option)
        return self.renderer.render()

    def close(self):
        for item in (self.renderer, self.viewer):
            if item is not None:
                item.close()
        self.renderer = self.viewer = None
