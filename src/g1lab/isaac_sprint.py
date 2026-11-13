"""G1 sprint task: flight-phase running on top of the v0-gain velocity task. Import only inside Isaac Sim.

Design (all numbers are initial hypotheses to be tuned by ablation, not validated settings):

* Speed gate s(v_cmd) = clip((v_cmd - v0) / (v1 - v0), 0, 1). Below v0 the reward equals the walking reward
  the public ONNX actor was trained with (so a transplanted actor is not disturbed); above v1 the walking-only
  terms are switched off and sprint terms are fully on.
* Sprint terms use contact-sensor events instead of a gait clock, because the policy does not observe a clock:
  flight phase, no double support, short stance, long swing, forward push-off force, forefoot touchdown,
  knee lift, ankle dorsiflexion in swing, contralateral arm swing, and a staged forward lean that grows with
  commanded speed and acceleration.
* The command ramps toward its sampled target with acceleration/deceleration limits, so acceleration,
  constant-speed and braking/stop phases all occur during training.
* Actuator limits, PD gains, observation and action spaces are unchanged (480 -> 29), so the public ONNX
  actor can still be transplanted exactly.
"""

import math
from collections.abc import Sequence
from dataclasses import MISSING

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
import torch
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat
from unitree_rl_lab.tasks.locomotion import mdp as vendor_mdp
from unitree_rl_lab.tasks.locomotion.mdp.commands import UniformLevelVelocityCommandCfg

from g1lab.isaac_v0_cfg import V0GainsEnvCfg

FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]
KNEES = ["left_knee_link", "right_knee_link"]
# Foot contact spheres of the G1 model (ankle_roll_link frame, metres): toe x=+0.12, heel x=-0.05, z=-0.03.
TOE_LOCAL = (0.12, 0.0, -0.03)
HEEL_LOCAL = (-0.05, 0.0, -0.03)
GATE = {"v0": 1.6, "v1": 2.2}
# Gait events use a feet-only sensor with a 20 N threshold (~6% body weight) so brief grazing contacts do not
# split one stance into several; the task's all-body sensor (1 N) still drives undesired-contact penalties.
FEET_SENSOR = "feet_contact"
FEET_FORCE_THRESHOLD_N = 20.0
RELEASE_STEPS = 2  # 40 ms without contact before a lift-off is declared


# ----------------------------------------------------------------------------------------------- command
class SprintVelocityCommand(UniformVelocityCommand):
    """Forward command that ramps toward a sampled target with bounded acceleration and deceleration."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.target_b = torch.zeros_like(self.vel_command_b)
        self.accel_x = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None):
        extras = super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.vel_command_b[ids] = 0.0  # every episode starts from rest and must accelerate
        self.accel_x[ids] = 0.0
        return extras

    def _resample_command(self, env_ids: Sequence[int]):
        current = self.vel_command_b[env_ids].clone()
        super()._resample_command(env_ids)
        self.target_b[env_ids] = self.vel_command_b[env_ids]
        self.vel_command_b[env_ids] = current

    def _update_command(self):
        target = self.target_b.clone()
        target[self.is_standing_env] = 0.0
        dt = self._env.step_dt
        delta = target[:, 0] - self.vel_command_b[:, 0]
        step = torch.clamp(delta, -self.cfg.decel_limit * dt, self.cfg.accel_limit * dt)
        self.vel_command_b[:, 0] += step
        self.vel_command_b[:, 1:] = target[:, 1:]
        self.accel_x = step / dt


@configclass
class SprintVelocityCommandCfg(UniformLevelVelocityCommandCfg):
    class_type: type = SprintVelocityCommand
    accel_limit: float = MISSING  # m/s^2
    decel_limit: float = MISSING  # m/s^2


# ----------------------------------------------------------------------------------------------- helpers
def _gate(env, command_name, v0, v1):
    v = env.command_manager.get_command(command_name)[:, 0]
    return torch.clamp((v - v0) / (v1 - v0), 0.0, 1.0)


class GaitTracker:
    """Debounced per-foot stance/swing state, updated once per control step.

    Touchdown is immediate (force above threshold in any physics substep of the step); lift-off is declared only
    after `RELEASE_STEPS` consecutive steps without contact, so grazing/rolling contacts do not split a stance.
    Durations are corrected for that delay. The landing posture (heel height minus toe height) is taken from the
    step before touchdown, i.e. how the foot approaches the ground.
    """

    def __init__(self, env, sensor_cfg, foot_cfg):
        ids = sensor_cfg.body_ids
        # SceneEntityCfg resolves "all bodies of the sensor" to slice(None); probe showed order [left, right].
        k = env.scene.sensors[sensor_cfg.name].num_bodies if isinstance(ids, slice) else len(ids)
        n, dev = env.num_envs, env.device
        self.sensor_cfg, self.foot_cfg = sensor_cfg, foot_cfg
        self.step = -1
        self.contact = torch.zeros(n, k, dtype=torch.bool, device=dev)
        self.gap = torch.zeros(n, k, dtype=torch.long, device=dev)
        self.timer = torch.zeros(n, k, device=dev)
        self.touchdown = torch.zeros_like(self.contact)
        self.liftoff = torch.zeros_like(self.contact)
        self.last_stance = torch.zeros(n, k, device=dev)
        self.last_air = torch.zeros(n, k, device=dev)
        self.prev_drop = torch.zeros(n, k, device=dev)
        self.landing_drop = torch.zeros(n, k, device=dev)

    def update(self, env):
        if self.step == env.common_step_counter:
            return self
        self.step = env.common_step_counter
        sensor = env.scene.sensors[self.sensor_cfg.name]
        history = sensor.data.net_forces_w_history[:, :, self.sensor_cfg.body_ids]
        raw = (torch.norm(history, dim=-1) > FEET_FORCE_THRESHOLD_N).any(dim=1)
        drop = (
            _foot_points(env, self.foot_cfg, HEEL_LOCAL)[..., 2]
            - _foot_points(env, self.foot_cfg, TOE_LOCAL)[..., 2]
        )
        dt = env.step_dt
        self.gap = torch.where(raw, torch.zeros_like(self.gap), self.gap + 1)
        contact = (self.contact | raw) & ~(self.contact & (self.gap >= RELEASE_STEPS))
        self.touchdown = contact & ~self.contact
        self.liftoff = self.contact & ~contact
        self.timer += dt
        delay = RELEASE_STEPS * dt
        self.last_stance = torch.where(self.liftoff, self.timer - delay, self.last_stance)
        self.last_air = torch.where(self.touchdown, self.timer, self.last_air)
        self.landing_drop = torch.where(self.touchdown, self.prev_drop, self.landing_drop)
        self.timer = torch.where(self.liftoff, torch.full_like(self.timer, delay), self.timer)
        self.timer = torch.where(self.touchdown, torch.zeros_like(self.timer), self.timer)
        self.contact = contact
        self.prev_drop = drop
        fresh = (env.episode_length_buf <= 1).unsqueeze(1).expand_as(contact)
        self.contact = torch.where(fresh, raw, self.contact)
        for buf in (self.touchdown, self.liftoff):
            buf[fresh] = False
        self.timer[fresh] = 0.0
        self.gap[fresh] = 0
        return self


def gait(env, sensor_cfg):
    tracker = env.__dict__.get("_g1lab_gait")
    if tracker is None:
        foot_cfg = SceneEntityCfg("robot", body_names=FEET, preserve_order=True)
        foot_cfg.resolve(env.scene)
        tracker = env.__dict__["_g1lab_gait"] = GaitTracker(env, sensor_cfg, foot_cfg)
    return tracker.update(env)


def _in_contact(env, sensor_cfg):
    return gait(env, sensor_cfg).contact


def _body_masses(env, asset_cfg):
    cache = env.__dict__.setdefault("_g1lab_body_masses", {})
    if asset_cfg.name not in cache:
        cache[asset_cfg.name] = env.scene[asset_cfg.name].root_physx_view.get_masses().to(env.device)
    return cache[asset_cfg.name]


def _foot_points(env, asset_cfg, local):
    asset = env.scene[asset_cfg.name]
    quat = asset.data.body_quat_w[:, asset_cfg.body_ids]
    offset = torch.tensor(local, device=env.device).expand(quat.shape[0], quat.shape[1], 3)
    return asset.data.body_pos_w[:, asset_cfg.body_ids] + quat_apply(quat, offset)


def torso_pitch(env, asset_cfg):
    """Forward lean of the torso (rad, positive = leaning forward) from its projected gravity."""
    asset = env.scene[asset_cfg.name]
    quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]
    gravity = torch.tensor((0.0, 0.0, -1.0), device=env.device).expand(quat.shape[0], 3)
    g_b = quat_apply_inverse(quat, gravity)
    return torch.asin(torch.clamp(g_b[:, 0], -1.0, 1.0)), g_b


# ------------------------------------------------------------------------------------------ sprint terms
def flight_phase(env, command_name, sensor_cfg, v0, v1):
    return _gate(env, command_name, v0, v1) * (~_in_contact(env, sensor_cfg)).all(dim=1).float()


def double_support(env, command_name, sensor_cfg, v0, v1):
    return _gate(env, command_name, v0, v1) * _in_contact(env, sensor_cfg).all(dim=1).float()


def stance_too_long(env, command_name, sensor_cfg, v0, v1, intercept, slope, minimum):
    """At each lift-off: seconds of stance beyond t_max(v) = clip(intercept - slope*v, minimum, intercept)."""
    g = gait(env, sensor_cfg)
    v = env.command_manager.get_command(command_name)[:, 0]
    t_max = torch.clamp(intercept - slope * v, minimum, intercept).unsqueeze(1)
    excess = torch.clamp(g.last_stance - t_max, min=0.0)
    return _gate(env, command_name, v0, v1) * torch.sum(excess * g.liftoff.float(), dim=1)


def swing_air_time(env, command_name, sensor_cfg, v0, v1, threshold, cap):
    """At each touchdown: (air time - threshold), capped; negative for too-short swings."""
    g = gait(env, sensor_cfg)
    air = torch.clamp(g.last_air, max=cap)
    return _gate(env, command_name, v0, v1) * torch.sum((air - threshold) * g.touchdown.float(), dim=1)


def push_off_force(env, command_name, sensor_cfg, asset_cfg, v0, v1):
    """Forward ground force while a foot is on the ground, from Newton: F = sum_i m_i a_i (heading frame).

    Isaac's contact-sensor net force carries only the normal component, so horizontal ground reaction force is
    reconstructed from body accelerations. Positive part only, in body weights.
    """
    asset = env.scene[asset_cfg.name]
    masses = _body_masses(env, asset_cfg)
    force_w = torch.sum(masses.unsqueeze(-1) * asset.data.body_lin_acc_w, dim=1)
    heading = yaw_quat(asset.data.root_quat_w)
    forward = quat_apply_inverse(heading, force_w)[:, 0]
    stance = _in_contact(env, sensor_cfg).any(dim=1).float()
    weight = masses.sum(dim=1) * 9.81
    return _gate(env, command_name, v0, v1) * stance * torch.clamp(forward, min=0.0) / weight


def forefoot_touchdown(env, command_name, sensor_cfg, asset_cfg, v0, v1, margin):
    """At each touchdown: +1 when the approaching toe is `margin` lower than the heel, -1 for a heel strike."""
    g = gait(env, sensor_cfg)
    score = torch.clamp(g.landing_drop / margin, -1.0, 1.0)
    return _gate(env, command_name, v0, v1) * torch.sum(score * g.touchdown.float(), dim=1)


def knee_lift(env, command_name, sensor_cfg, asset_cfg, v0, v1, standing_drop, target_drop):
    """Swing legs: knee raised from its standing depth below the pelvis toward `target_drop`."""
    asset = env.scene[asset_cfg.name]
    drop = asset.data.root_pos_w[:, 2:3] - asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    lift = torch.clamp((standing_drop - drop) / (standing_drop - target_drop), 0.0, 1.0)
    swing = (~_in_contact(env, sensor_cfg)).float()
    return _gate(env, command_name, v0, v1) * torch.sum(lift * swing, dim=1)


def swing_dorsiflexion(env, command_name, sensor_cfg, asset_cfg, foot_cfg, v0, v1, min_foot_height, span):
    """Swing feet above `min_foot_height`: ankle pitched toe-up (negative pitch) relative to default."""
    asset = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    q0 = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    dorsi = torch.clamp((q0 - q) / span, 0.0, 1.0)
    high = env.scene[foot_cfg.name].data.body_pos_w[:, foot_cfg.body_ids, 2] > min_foot_height
    swing = (~_in_contact(env, sensor_cfg)) & high
    return _gate(env, command_name, v0, v1) * torch.sum(dorsi * swing.float(), dim=1)


def arm_leg_counter_swing(env, command_name, shoulder_cfg, hip_cfg, v0, v1, scale):
    """Arm swings opposite to the same-side thigh: rewards -dq_shoulder_pitch * dq_hip_pitch (both sides)."""
    asset = env.scene[shoulder_cfg.name]
    product = asset.data.joint_vel[:, shoulder_cfg.joint_ids] * asset.data.joint_vel[:, hip_cfg.joint_ids]
    return _gate(env, command_name, v0, v1) * torch.tanh(-product.sum(dim=1) / scale)


def staged_lean(env, command_name, asset_cfg, v0, v1, base, per_speed, per_accel, accel_range, band):
    """Squared torso-pitch error outside a band around base + per_speed*v + per_accel*a (rad)."""
    pitch, _ = torso_pitch(env, asset_cfg)
    term = env.command_manager.get_term(command_name)
    accel = torch.clamp(term.accel_x, *accel_range)
    target = base + per_speed * term.command[:, 0] + per_accel * accel
    error = torch.clamp(torch.abs(pitch - target) - band, min=0.0)
    return _gate(env, command_name, v0, v1) * torch.square(error)


def torso_roll_l2(env, command_name, asset_cfg, v0, v1):
    _, g_b = torso_pitch(env, asset_cfg)
    return _gate(env, command_name, v0, v1) * torch.square(g_b[:, 1])


# ------------------------------------------------------------------- walking terms faded out when running
def _walk(env, command_name, v0, v1):
    return 1.0 - _gate(env, command_name, v0, v1)


def walk_lin_vel_z_l2(env, command_name, v0, v1, asset_cfg=SceneEntityCfg("robot")):
    return _walk(env, command_name, v0, v1) * vendor_mdp.lin_vel_z_l2(env, asset_cfg)


def walk_flat_orientation_l2(env, command_name, v0, v1, asset_cfg=SceneEntityCfg("robot")):
    return _walk(env, command_name, v0, v1) * vendor_mdp.flat_orientation_l2(env, asset_cfg)


def walk_base_height_l2(env, command_name, v0, v1, target_height, asset_cfg=SceneEntityCfg("robot")):
    return _walk(env, command_name, v0, v1) * vendor_mdp.base_height_l2(env, target_height, asset_cfg)


def walk_joint_deviation_l1(env, command_name, v0, v1, asset_cfg):
    return _walk(env, command_name, v0, v1) * vendor_mdp.joint_deviation_l1(env, asset_cfg)


def run_joint_deviation_l1(env, command_name, v0, v1, asset_cfg):
    return _gate(env, command_name, v0, v1) * vendor_mdp.joint_deviation_l1(env, asset_cfg)


def walk_feet_gait(env, command_name, v0, v1, period, offset, sensor_cfg, threshold):
    return _walk(env, command_name, v0, v1) * vendor_mdp.feet_gait(
        env, period, offset, sensor_cfg, threshold, command_name
    )


# --------------------------------------------------------------------------------------------- config
def _feet_sensor():
    return SceneEntityCfg(FEET_SENSOR, body_names=FEET, preserve_order=True)


def _feet_body():
    return SceneEntityCfg("robot", body_names=FEET, preserve_order=True)


def _sides(pattern):
    return SceneEntityCfg("robot", joint_names=[f"left_{pattern}", f"right_{pattern}"], preserve_order=True)


SPRINT_WEIGHTS = {
    "flight_phase": 2.0,
    "double_support": -1.0,
    "stance_too_long": -10.0,
    "swing_air_time": 2.0,
    "push_off_force": 0.5,
    "forefoot_touchdown": 1.0,
    "knee_lift": 0.5,
    "swing_dorsiflexion": 0.3,
    "arm_leg_counter_swing": 0.5,
    "staged_lean": -20.0,
    "torso_roll": -5.0,
    "run_joint_deviation_arms": -0.1,
}


@configclass
class SprintEnvCfg(V0GainsEnvCfg):
    sprint_speed_limit: float = 3.5

    def __post_init__(self):
        super().__post_init__()
        # Offline-safe, long flat ground. The default "plane" terrain loads default_environment.usd from the
        # NVIDIA S3 nucleus; the sky HDR, terrain MDL and command arrows are also remote. None are needed for
        # headless training, so the ground is a locally generated flat mesh: 3x3 tiles of 250 m (+50 m border),
        # enough for 20 s episodes at 11 m/s. Envs sharing a tile overlap but inter-env collisions are filtered.
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=terrain_gen.TerrainGeneratorCfg(
                size=(250.0, 250.0),
                border_width=50.0,
                num_rows=3,
                num_cols=3,
                horizontal_scale=0.5,
                vertical_scale=0.005,
                slope_threshold=None,
                use_cache=False,
                curriculum=False,
                sub_terrains={"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)},
            ),
            max_init_terrain_level=None,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=None,
            debug_vis=False,
        )
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.sky_light = None
        self.scene.height_scanner = None
        self.curriculum.terrain_levels = None
        self.scene.feet_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link",
            history_length=self.decimation,  # every physics substep of a control step
            track_air_time=True,
            force_threshold=FEET_FORCE_THRESHOLD_N,
            update_period=self.sim.dt,
        )
        cmd = "base_velocity"
        gate = {"command_name": cmd, **GATE}

        self.commands.base_velocity = SprintVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 8.0),
            rel_standing_envs=0.1,
            rel_heading_envs=0.0,
            heading_command=False,
            debug_vis=False,  # arrow markers are remote assets
            ranges=SprintVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 1.5), lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.2, 0.2)
            ),
            limit_ranges=SprintVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, self.sprint_speed_limit), lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.2, 0.2)
            ),
            accel_limit=2.0,
            decel_limit=2.5,
        )
        self.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)

        r = self.rewards
        # Walking-regime terms keep their original weights but fade out as the command enters running speeds.
        r.base_linear_velocity = RewTerm(func=walk_lin_vel_z_l2, weight=-2.0, params=dict(gate))
        r.flat_orientation_l2 = RewTerm(func=walk_flat_orientation_l2, weight=-5.0, params=dict(gate))
        r.base_height = RewTerm(
            func=walk_base_height_l2, weight=-10.0, params={**gate, "target_height": 0.78}
        )
        arms = r.joint_deviation_arms.params["asset_cfg"]
        r.joint_deviation_arms = RewTerm(
            func=walk_joint_deviation_l1, weight=-0.1, params={**gate, "asset_cfg": arms}
        )
        g = r.gait.params
        r.gait = RewTerm(
            func=walk_feet_gait,
            weight=0.5,
            params={
                **gate,
                "period": g["period"],
                "offset": g["offset"],
                "sensor_cfg": g["sensor_cfg"],
                "threshold": g["threshold"],
            },
        )

        w = SPRINT_WEIGHTS
        feet = _feet_sensor()
        r.flight_phase = RewTerm(
            func=flight_phase, weight=w["flight_phase"], params={**gate, "sensor_cfg": feet}
        )
        r.double_support = RewTerm(
            func=double_support, weight=w["double_support"], params={**gate, "sensor_cfg": _feet_sensor()}
        )
        r.stance_too_long = RewTerm(
            func=stance_too_long,
            weight=w["stance_too_long"],
            params={**gate, "sensor_cfg": _feet_sensor(), "intercept": 0.36, "slope": 0.06, "minimum": 0.15},
        )
        r.swing_air_time = RewTerm(
            func=swing_air_time,
            weight=w["swing_air_time"],
            params={**gate, "sensor_cfg": _feet_sensor(), "threshold": 0.30, "cap": 0.55},
        )
        r.push_off_force = RewTerm(
            func=push_off_force,
            weight=w["push_off_force"],
            params={**gate, "sensor_cfg": _feet_sensor(), "asset_cfg": SceneEntityCfg("robot")},
        )
        r.forefoot_touchdown = RewTerm(
            func=forefoot_touchdown,
            weight=w["forefoot_touchdown"],
            params={**gate, "sensor_cfg": _feet_sensor(), "asset_cfg": _feet_body(), "margin": 0.02},
        )
        r.knee_lift = RewTerm(
            func=knee_lift,
            weight=w["knee_lift"],
            params={
                **gate,
                "sensor_cfg": _feet_sensor(),
                "asset_cfg": SceneEntityCfg("robot", body_names=KNEES, preserve_order=True),
                "standing_drop": 0.44,  # measured in PhysX at the default pose
                "target_drop": 0.22,  # thigh roughly 70 deg forward of vertical
            },
        )
        r.swing_dorsiflexion = RewTerm(
            func=swing_dorsiflexion,
            weight=w["swing_dorsiflexion"],
            params={
                **gate,
                "sensor_cfg": _feet_sensor(),
                "asset_cfg": _sides("ankle_pitch_joint"),
                "foot_cfg": _feet_body(),
                "min_foot_height": 0.08,
                "span": 0.3,
            },
        )
        r.arm_leg_counter_swing = RewTerm(
            func=arm_leg_counter_swing,
            weight=w["arm_leg_counter_swing"],
            params={
                **gate,
                "shoulder_cfg": _sides("shoulder_pitch_joint"),
                "hip_cfg": _sides("hip_pitch_joint"),
                "scale": 20.0,
            },
        )
        torso = SceneEntityCfg("robot", body_names=["torso_link"])
        r.staged_lean = RewTerm(
            func=staged_lean,
            weight=w["staged_lean"],
            params={
                **gate,
                "asset_cfg": torso,
                "base": 0.05,
                "per_speed": 0.04,
                "per_accel": 0.06,
                "accel_range": (-1.5, 2.0),
                "band": 0.05,
            },
        )
        r.torso_roll = RewTerm(
            func=torso_roll_l2,
            weight=w["torso_roll"],
            params={**gate, "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"])},
        )
        r.run_joint_deviation_arms = RewTerm(
            func=run_joint_deviation_l1,
            weight=w["run_joint_deviation_arms"],
            params={
                **gate,
                # shoulder pitch is left free for arm swing; elbow stays near its bent default
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        ".*_shoulder_roll_joint",
                        ".*_shoulder_yaw_joint",
                        ".*_elbow_joint",
                        ".*_wrist_.*",
                    ],
                ),
            },
        )


def lean_target_deg(speed, accel):
    """Documented staged-lean target for reports (deg)."""
    a = min(max(accel, -1.5), 2.0)
    return math.degrees(0.05 + 0.04 * speed + 0.06 * a)
