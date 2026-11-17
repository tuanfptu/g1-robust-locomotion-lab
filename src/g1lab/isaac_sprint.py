"""G1 sprint task (reward design v2): flight-phase running within the actuator spec. Import only inside Isaac Sim.

Design principles (hypotheses, to be checked by ablation and the MuJoCo benchmark):

* Speed comes from **step frequency** with a moderate step length, not from long hops: a cadence target that rises
  with speed replaces "maximise air time"; a flight phase must exist (small bonus, no double support) but is not
  maximised.
* Feet land close under the centre of mass with little forward foot speed (less braking); propulsion is rewarded
  only while accelerating, braking force is penalised at constant speed.
* Mid/forefoot landing (not heel); ankle dorsiflexed and thigh raised in swing.
* Body twisting is attacked at its source: whole-body angular momentum about the vertical axis, narrow step width,
  and a bounded heading-tracking bonus.
* Forward lean follows commanded acceleration (atan(a/g) + 4 deg), not speed.
* Stay inside the actuator envelope (g1lab.actuator_spec): penalties for torque clipping and joint speed above 90 %
  of the rated speed.
* All running terms are gated by command speed s(v) = clip((v - 1.6) / 0.6, 0, 1); below that the vendor walking
  reward applies. Bonuses are bounded and non-negative, and falling costs -200, so ending an episode early is never
  a way to escape penalties.
* The command ramps with bounded acceleration and holds a sampled heading, so acceleration, cruise, braking and stop
  all occur in training.
"""

from collections.abc import Sequence
from dataclasses import MISSING

import isaaclab.envs.mdp as lab_mdp
import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse, wrap_to_pi, yaw_quat
from unitree_rl_lab.assets.robots.unitree_actuators import UnitreeActuatorCfg
from unitree_rl_lab.tasks.locomotion import mdp as vendor_mdp
from unitree_rl_lab.tasks.locomotion.mdp.commands import UniformLevelVelocityCommandCfg

from g1lab.actuator_spec import SPEC
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
# m^2/s. Probe of the previous best policy (Isaac, 1-5 m/s): whole-body L_z/m RMS 0.021-0.027 even while the pelvis
# yaw-rate RMS was 0.8-1.1 rad/s, i.e. the pelvis counter-rotates against the legs; pelvis yaw is handled by
# run_yaw_tracking, this term only keeps whole-body twist small.
LOW_TWIST_STD = 0.02


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
        if self.cfg.heading_command:
            # Hold a sampled heading (like the MuJoCo benchmark's heading hold): yaw-rate command from heading error.
            ids = self.is_heading_env & ~self.is_standing_env
            error = wrap_to_pi(self.heading_target - self.robot.data.heading_w)
            yaw = torch.clip(self.cfg.heading_control_stiffness * error, *self.cfg.ranges.ang_vel_z)
            self.vel_command_b[:, 2] = torch.where(ids, yaw, self.vel_command_b[:, 2])


@configclass
class SprintVelocityCommandCfg(UniformLevelVelocityCommandCfg):
    class_type: type = SprintVelocityCommand
    accel_limit: float = MISSING  # m/s^2
    decel_limit: float = MISSING  # m/s^2


# ----------------------------------------------------------------------------------------------- helpers
def _gate(env, command_name, v0, v1):
    v = env.command_manager.get_command(command_name)[:, 0]
    return torch.clamp((v - v0) / (v1 - v0), 0.0, 1.0)


EARNED_STD = (0.5, 0.15)  # same tolerance as the speed-tracking reward


def _earned(env, command_name):
    """Speed-tracking quality in [0, 1] (same kernel as track_lin_vel_scaled).

    Positive running-style bonuses are paid only in proportion to it. Without this, the policy of run 2026-11-16
    learned to ignore fast commands and jog straight at ~2 m/s: heading/twist/knee bonuses were gated by the
    *commanded* speed only, so they paid in full while running far slower than asked (rounds 18-30: speed
    tracking reward 1.12 -> 0.41 while yaw bonuses 0.79 -> 1.56 and the Isaac top speed fell 5.4 -> 2.0 m/s).
    Penalties are not scaled, so slowing down never makes them cheaper.
    """
    floor = float(getattr(env.cfg, "earned_floor", 0.0))  # 1.0 = reward v2 (bonuses ignore speed tracking)
    q = track_lin_vel_scaled(env, command_name, *EARNED_STD)
    return floor + (1.0 - floor) * q


class GaitTracker:
    """Debounced per-foot stance/swing state and touchdown kinematics, updated once per control step.

    Touchdown is immediate (force above threshold in any physics substep of the step); lift-off is declared only
    after `RELEASE_STEPS` consecutive steps without contact, so grazing/rolling contacts do not split a stance.
    At each touchdown it records: the landing posture (heel minus toe height, taken one step before contact), the
    step time since the previous touchdown of either foot and whether feet alternated, the foot position relative
    to the whole-body centre of mass in the heading frame (ahead / lateral), and the forward foot speed just before
    contact.
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
        self.prev_foot_vx = torch.zeros(n, k, device=dev)
        self.landing_foot_vx = torch.zeros(n, k, device=dev)
        self.landing_ahead = torch.zeros(n, k, device=dev)
        self.landing_lateral = torch.zeros(n, k, device=dev)
        self.since_touchdown = torch.zeros(n, device=dev)
        self.step_time = torch.zeros(n, device=dev)
        self.last_foot = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.alternated = torch.zeros(n, dtype=torch.bool, device=dev)
        self.any_touchdown = torch.zeros(n, dtype=torch.bool, device=dev)

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
        robot = env.scene[self.foot_cfg.name]
        heading = yaw_quat(robot.data.root_quat_w).unsqueeze(1).expand(-1, drop.shape[1], -1)
        com = _com(env)
        rel = quat_apply_inverse(heading, robot.data.body_pos_w[:, self.foot_cfg.body_ids] - com.unsqueeze(1))
        foot_v = quat_apply_inverse(heading, robot.data.body_lin_vel_w[:, self.foot_cfg.body_ids])
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
        self.landing_foot_vx = torch.where(self.touchdown, self.prev_foot_vx, self.landing_foot_vx)
        self.landing_ahead = torch.where(self.touchdown, rel[..., 0], self.landing_ahead)
        self.landing_lateral = torch.where(self.touchdown, rel[..., 1].abs(), self.landing_lateral)
        self.timer = torch.where(self.liftoff, torch.full_like(self.timer, delay), self.timer)
        self.timer = torch.where(self.touchdown, torch.zeros_like(self.timer), self.timer)
        # step timing / alternation (one touchdown per step; if both land together count it as a non-alternation)
        self.since_touchdown += dt
        self.any_touchdown = self.touchdown.any(dim=1)
        foot = torch.argmax(self.touchdown.float(), dim=1)
        both = self.touchdown.all(dim=1)
        self.alternated = self.any_touchdown & ~both & (self.last_foot >= 0) & (foot != self.last_foot)
        self.step_time = torch.where(self.any_touchdown, self.since_touchdown, self.step_time)
        self.since_touchdown = torch.where(
            self.any_touchdown, torch.zeros_like(self.since_touchdown), self.since_touchdown
        )
        self.last_foot = torch.where(self.any_touchdown, foot, self.last_foot)
        self.contact = contact
        self.prev_drop = drop
        self.prev_foot_vx = foot_v[..., 0]
        fresh_env = env.episode_length_buf <= 1
        fresh = fresh_env.unsqueeze(1).expand_as(contact)
        self.contact = torch.where(fresh, raw, self.contact)
        for buf in (self.touchdown, self.liftoff):
            buf[fresh] = False
        self.timer[fresh] = 0.0
        self.gap[fresh] = 0
        self.any_touchdown[fresh_env] = False
        self.alternated[fresh_env] = False
        self.last_foot[fresh_env] = -1
        self.since_touchdown[fresh_env] = 0.0
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


def _com(env, asset_name="robot"):
    """Whole-body centre of mass in world frame (uses the possibly randomised masses)."""
    masses = _body_masses(env, SceneEntityCfg(asset_name))
    pos = env.scene[asset_name].data.body_com_pos_w
    return torch.sum(masses.unsqueeze(-1) * pos, dim=1) / masses.sum(dim=1, keepdim=True)


def vertical_angular_momentum_per_mass(env, asset_name="robot"):
    """Whole-body angular momentum about the vertical axis through the CoM, divided by total mass (m^2/s).

    Orbital part sum m_i ((p_i - c) x (v_i - v_c))_z plus an approximate spin part sum I_zz,i w_z,i (body-frame zz
    inertia, ignoring off-diagonal terms) - enough to detect the body twisting that failed the yaw criterion.
    """
    asset = env.scene[asset_name]
    masses = _body_masses(env, SceneEntityCfg(asset_name))
    cache = env.__dict__.setdefault("_g1lab_inertia_zz", {})
    if asset_name not in cache:
        cache[asset_name] = asset.root_physx_view.get_inertias().to(env.device)[..., 8]
    total = masses.sum(dim=1, keepdim=True)
    pos, vel = asset.data.body_com_pos_w, asset.data.body_com_lin_vel_w
    c = torch.sum(masses.unsqueeze(-1) * pos, dim=1, keepdim=True) / total.unsqueeze(-1)
    vc = torch.sum(masses.unsqueeze(-1) * vel, dim=1, keepdim=True) / total.unsqueeze(-1)
    r, v = pos - c, vel - vc
    orbital = torch.sum(masses * (r[..., 0] * v[..., 1] - r[..., 1] * v[..., 0]), dim=1)
    spin = torch.sum(cache[asset_name] * asset.data.body_ang_vel_w[..., 2], dim=1)
    return (orbital + spin) / total.squeeze(1)


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
def track_lin_vel_scaled(env, command_name, base_std, rel_std, asset_cfg=SceneEntityCfg("robot")):
    """exp(-|v_cmd - v|^2 / sigma^2) in the yaw frame with sigma = max(base_std, rel_std * |v_cmd|).

    Same name/slot as the vendor term (the speed curriculum reads it); the tolerance grows with speed so a 5 % error
    at 8 m/s is not treated like a 0.4 m/s error at walking speed.
    """
    asset = env.scene[asset_cfg.name]
    vel = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    cmd = env.command_manager.get_command(command_name)
    error = torch.sum(torch.square(cmd[:, :2] - vel[:, :2]), dim=1)
    sigma = torch.clamp(rel_std * torch.norm(cmd[:, :2], dim=1), min=base_std)
    return torch.exp(-error / sigma**2)


def flight_phase(env, command_name, sensor_cfg, v0, v1):
    return (
        _gate(env, command_name, v0, v1)
        * _earned(env, command_name)
        * (~_in_contact(env, sensor_cfg)).all(dim=1).float()
    )


def double_support(env, command_name, sensor_cfg, v0, v1):
    return _gate(env, command_name, v0, v1) * _in_contact(env, sensor_cfg).all(dim=1).float()


def cadence_tracking(env, command_name, sensor_cfg, v0, v1, base, per_speed, low, high, std):
    """At each alternating touchdown: exp(-((1/step_time - f*(v)) / std)^2), f* = clip(base + per_speed*v, low, high).

    Rewards turnover instead of long aerial hops; a non-alternating touchdown (hop, both feet) earns nothing.
    """
    g = gait(env, sensor_cfg)
    v = env.command_manager.get_command(command_name)[:, 0]
    target = torch.clamp(base + per_speed * v, low, high)
    freq = 1.0 / torch.clamp(g.step_time, min=1e-3)
    score = torch.exp(-torch.square((freq - target) / std)) * g.alternated.float()
    return _gate(env, command_name, v0, v1) * _earned(env, command_name) * score


def stance_too_long(env, command_name, sensor_cfg, v0, v1, intercept, slope, minimum):
    """At each lift-off: seconds of stance beyond t_max(v) = clip(intercept - slope*v, minimum, intercept)."""
    g = gait(env, sensor_cfg)
    v = env.command_manager.get_command(command_name)[:, 0]
    t_max = torch.clamp(intercept - slope * v, minimum, intercept).unsqueeze(1)
    excess = torch.clamp(g.last_stance - t_max, min=0.0)
    return _gate(env, command_name, v0, v1) * torch.sum(excess * g.liftoff.float(), dim=1)


def overstride(env, command_name, sensor_cfg, v0, v1, max_ahead, scale):
    """At each touchdown: how far the foot lands beyond `max_ahead` in front of the CoM (braking), in units of scale."""
    g = gait(env, sensor_cfg)
    excess = torch.clamp(g.landing_ahead - max_ahead, min=0.0) / scale
    return _gate(env, command_name, v0, v1) * torch.sum(excess * g.touchdown.float(), dim=1)


def landing_foot_speed(env, command_name, sensor_cfg, v0, v1):
    """At each touchdown: forward foot speed just before contact relative to the command speed (no 'pawing')."""
    g = gait(env, sensor_cfg)
    v = torch.clamp(env.command_manager.get_command(command_name)[:, 0], min=1.0).unsqueeze(1)
    excess = torch.clamp(g.landing_foot_vx, min=0.0) / v
    return _gate(env, command_name, v0, v1) * torch.sum(excess * g.touchdown.float(), dim=1)


def wide_step(env, command_name, sensor_cfg, v0, v1, max_lateral, scale):
    """At each touchdown: lateral foot distance from the CoM beyond `max_lateral` (wide steps create yaw torque)."""
    g = gait(env, sensor_cfg)
    excess = torch.clamp(g.landing_lateral - max_lateral, min=0.0) / scale
    return _gate(env, command_name, v0, v1) * torch.sum(excess * g.touchdown.float(), dim=1)


def _forward_ground_force(env, asset_cfg):
    asset = env.scene[asset_cfg.name]
    masses = _body_masses(env, asset_cfg)
    force_w = torch.sum(masses.unsqueeze(-1) * asset.data.body_lin_acc_w, dim=1)
    forward = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), force_w)[:, 0]
    return forward / (masses.sum(dim=1) * 9.81)


def accel_push_off(env, command_name, sensor_cfg, asset_cfg, v0, v1, accel_scale):
    """Forward ground force (body weights, Newton: sum m_i a_i) during stance, only while the command accelerates."""
    term = env.command_manager.get_term(command_name)
    accelerating = torch.clamp(term.accel_x / accel_scale, 0.0, 1.0)
    stance = _in_contact(env, sensor_cfg).any(dim=1).float()
    push = torch.clamp(_forward_ground_force(env, asset_cfg), min=0.0)
    return _gate(env, command_name, v0, v1) * _earned(env, command_name) * accelerating * stance * push


def cruise_braking(env, command_name, sensor_cfg, asset_cfg, v0, v1, accel_scale):
    """Backward ground force (body weights) during stance while not commanded to slow down."""
    term = env.command_manager.get_term(command_name)
    not_braking = 1.0 - torch.clamp(-term.accel_x / accel_scale, 0.0, 1.0)
    stance = _in_contact(env, sensor_cfg).any(dim=1).float()
    brake = torch.clamp(-_forward_ground_force(env, asset_cfg), min=0.0)
    return _gate(env, command_name, v0, v1) * not_braking * stance * brake


def midfoot_landing(env, command_name, sensor_cfg, asset_cfg, v0, v1, flat_tol, span):
    """At each touchdown: +1 forefoot (toe lower), about +0.3 flat, -1 heel strike (heel lower by > span - tol)."""
    g = gait(env, sensor_cfg)
    score = torch.clamp((g.landing_drop + flat_tol) / span, -1.0, 1.0)
    score = torch.where(
        score > 0, score * _earned(env, command_name).unsqueeze(1), score
    )  # heel penalty unscaled
    return _gate(env, command_name, v0, v1) * torch.sum(score * g.touchdown.float(), dim=1)


def knee_lift(env, command_name, sensor_cfg, asset_cfg, v0, v1, standing_drop, target_drop):
    """Swing legs: knee raised from its standing depth below the pelvis toward `target_drop`."""
    asset = env.scene[asset_cfg.name]
    drop = asset.data.root_pos_w[:, 2:3] - asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    lift = torch.clamp((standing_drop - drop) / (standing_drop - target_drop), 0.0, 1.0)
    swing = (~_in_contact(env, sensor_cfg)).float()
    return _gate(env, command_name, v0, v1) * _earned(env, command_name) * torch.sum(lift * swing, dim=1)


def swing_dorsiflexion(env, command_name, sensor_cfg, asset_cfg, foot_cfg, v0, v1, min_foot_height, span):
    """Swing feet above `min_foot_height`: ankle pitched toe-up (negative pitch) relative to default."""
    asset = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    q0 = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    dorsi = torch.clamp((q0 - q) / span, 0.0, 1.0)
    high = env.scene[foot_cfg.name].data.body_pos_w[:, foot_cfg.body_ids, 2] > min_foot_height
    swing = (~_in_contact(env, sensor_cfg)) & high
    return (
        _gate(env, command_name, v0, v1)
        * _earned(env, command_name)
        * torch.sum(dorsi * swing.float(), dim=1)
    )


def low_twist(env, command_name, v0, v1, std):
    """Bounded bonus exp(-(L_z / m / std)^2): whole-body angular momentum about the vertical stays small."""
    lz = vertical_angular_momentum_per_mass(env)
    return _gate(env, command_name, v0, v1) * _earned(env, command_name) * torch.exp(-torch.square(lz / std))


def acceleration_lean(env, command_name, asset_cfg, v0, v1, offset, band, accel_range):
    """Squared torso-pitch error outside a band around atan(a_cmd / g) + offset (rad)."""
    pitch, _ = torso_pitch(env, asset_cfg)
    term = env.command_manager.get_term(command_name)
    accel = torch.clamp(term.accel_x, *accel_range)
    target = torch.atan(accel / 9.81) + offset
    error = torch.clamp(torch.abs(pitch - target) - band, min=0.0)
    return _gate(env, command_name, v0, v1) * torch.square(error)


def run_yaw_tracking(env, command_name, v0, v1, std, asset_cfg=SceneEntityCfg("robot")):
    """Bounded, non-negative yaw-rate tracking bonus while running: exp(-error^2 / std^2).

    A squared-error penalty here made dying early the best option (episode length collapsed 618 -> 58 steps in
    the first spec run), so body twisting is discouraged by a bonus that only rewards staying on heading.
    """
    wz = env.scene[asset_cfg.name].data.root_ang_vel_b[:, 2]
    error = wz - env.command_manager.get_command(command_name)[:, 2]
    return (
        _gate(env, command_name, v0, v1)
        * _earned(env, command_name)
        * torch.exp(-torch.square(error) / std**2)
    )


def torso_roll_l2(env, command_name, asset_cfg, v0, v1):
    _, g_b = torso_pitch(env, asset_cfg)
    return _gate(env, command_name, v0, v1) * torch.square(g_b[:, 1])


def _spec_vector(env, asset, key):
    cache = env.__dict__.setdefault("_g1lab_spec", {})
    if key not in cache:
        cache[key] = torch.tensor([SPEC[name][key] for name in asset.joint_names], device=env.device)
    return cache[key]


def joint_speed_margin(env, threshold, asset_cfg=SceneEntityCfg("robot")):
    """Sum over joints of how far |qd| exceeds `threshold` x rated speed (actuator spec), all speeds."""
    asset = env.scene[asset_cfg.name]
    ratio = asset.data.joint_vel.abs() / _spec_vector(env, asset, "vel")
    return torch.sum(torch.clamp(ratio - threshold, min=0.0), dim=1)


def torque_clipping(env, tolerance, asset_cfg=SceneEntityCfg("robot")):
    """Sum over joints of requested-minus-delivered torque (beyond `tolerance` N*m, absorbs motor friction),
    relative to peak torque: time spent pushing the actuators past their envelope."""
    asset = env.scene[asset_cfg.name]
    excess = torch.clamp(
        asset.data.computed_torque.abs() - asset.data.applied_torque.abs() - tolerance, min=0.0
    )
    return torch.sum(excess / _spec_vector(env, asset, "y1"), dim=1)


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
    # task
    "track_lin_vel_xy": 1.5,
    "run_yaw_tracking": 2.0,
    "termination": -200.0,
    # gait (running only)
    "flight_phase": 0.5,
    "double_support": -1.0,
    "cadence_tracking": 2.0,
    "stance_too_long": -10.0,
    "overstride": -1.0,
    "landing_foot_speed": -1.0,
    "wide_step": -1.0,
    "accel_push_off": 1.0,
    "cruise_braking": -0.5,
    "midfoot_landing": 0.5,
    "knee_lift": 0.5,
    "swing_dorsiflexion": 0.3,
    # posture / twist (running only)
    "low_twist": 0.5,
    "acceleration_lean": -20.0,
    "torso_roll": -5.0,
    "run_joint_deviation_arms": -0.1,
    # actuator envelope (all speeds)
    "joint_speed_margin": -2.0,
    "torque_clipping": -1.0,
}


@configclass
class SprintEnvCfg(V0GainsEnvCfg):
    sprint_speed_limit: float = 3.5
    # Share of the running-style bonuses paid regardless of speed tracking (0 = reward v3, 1 = reward v2).
    earned_floor: float = 0.0

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
            rel_heading_envs=1.0,
            heading_command=True,
            heading_control_stiffness=1.0,
            debug_vis=False,  # arrow markers are remote assets
            ranges=SprintVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 1.5), lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.5, 0.5), heading=(-0.5, 0.5)
            ),
            limit_ranges=SprintVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, self.sprint_speed_limit),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-0.5, 0.5),
                heading=(-0.5, 0.5),
            ),
            accel_limit=2.0,
            decel_limit=2.5,
        )
        self.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)

        r = self.rewards
        w = SPRINT_WEIGHTS
        feet = _feet_sensor
        # Task: speed tracking with speed-relative tolerance (keeps the vendor slot the curriculum reads).
        r.track_lin_vel_xy = RewTerm(
            func=track_lin_vel_scaled,
            weight=w["track_lin_vel_xy"],
            params={"command_name": cmd, "base_std": 0.5, "rel_std": 0.15},
        )
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
        # Running gait.
        r.flight_phase = RewTerm(
            func=flight_phase, weight=w["flight_phase"], params={**gate, "sensor_cfg": feet()}
        )
        r.double_support = RewTerm(
            func=double_support, weight=w["double_support"], params={**gate, "sensor_cfg": feet()}
        )
        r.cadence_tracking = RewTerm(
            func=cadence_tracking,
            weight=w["cadence_tracking"],
            params={
                **gate,
                "sensor_cfg": feet(),
                "base": 3.0,
                "per_speed": 0.3,
                "low": 3.2,
                "high": 5.0,
                "std": 0.6,
            },
        )
        r.stance_too_long = RewTerm(
            func=stance_too_long,
            weight=w["stance_too_long"],
            params={**gate, "sensor_cfg": feet(), "intercept": 0.36, "slope": 0.04, "minimum": 0.12},
        )
        r.overstride = RewTerm(
            func=overstride,
            weight=w["overstride"],
            params={**gate, "sensor_cfg": feet(), "max_ahead": 0.15, "scale": 0.10},
        )
        r.landing_foot_speed = RewTerm(
            func=landing_foot_speed, weight=w["landing_foot_speed"], params={**gate, "sensor_cfg": feet()}
        )
        r.wide_step = RewTerm(
            func=wide_step,
            weight=w["wide_step"],
            params={**gate, "sensor_cfg": feet(), "max_lateral": 0.10, "scale": 0.05},
        )
        r.accel_push_off = RewTerm(
            func=accel_push_off,
            weight=w["accel_push_off"],
            params={**gate, "sensor_cfg": feet(), "asset_cfg": SceneEntityCfg("robot"), "accel_scale": 2.0},
        )
        r.cruise_braking = RewTerm(
            func=cruise_braking,
            weight=w["cruise_braking"],
            params={**gate, "sensor_cfg": feet(), "asset_cfg": SceneEntityCfg("robot"), "accel_scale": 2.0},
        )
        r.midfoot_landing = RewTerm(
            func=midfoot_landing,
            weight=w["midfoot_landing"],
            params={
                **gate,
                "sensor_cfg": feet(),
                "asset_cfg": _feet_body(),
                "flat_tol": 0.005,
                "span": 0.015,
            },
        )
        r.knee_lift = RewTerm(
            func=knee_lift,
            weight=w["knee_lift"],
            params={
                **gate,
                "sensor_cfg": feet(),
                "asset_cfg": SceneEntityCfg("robot", body_names=KNEES, preserve_order=True),
                "standing_drop": 0.44,  # measured in PhysX at the default pose
                "target_drop": 0.24,
            },
        )
        r.swing_dorsiflexion = RewTerm(
            func=swing_dorsiflexion,
            weight=w["swing_dorsiflexion"],
            params={
                **gate,
                "sensor_cfg": feet(),
                "asset_cfg": _sides("ankle_pitch_joint"),
                "foot_cfg": _feet_body(),
                "min_foot_height": 0.08,
                "span": 0.3,
            },
        )
        # Posture and twist.
        torso = SceneEntityCfg("robot", body_names=["torso_link"])
        r.low_twist = RewTerm(func=low_twist, weight=w["low_twist"], params={**gate, "std": LOW_TWIST_STD})
        r.acceleration_lean = RewTerm(
            func=acceleration_lean,
            weight=w["acceleration_lean"],
            params={**gate, "asset_cfg": torso, "offset": 0.07, "band": 0.05, "accel_range": (-3.0, 3.0)},
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
                # shoulder pitch free for fore-aft arm swing; lateral arm motion and elbow/wrist stay near default
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
        r.run_yaw_tracking = RewTerm(
            func=run_yaw_tracking, weight=w["run_yaw_tracking"], params={**gate, "std": 0.5}
        )
        # Actuator envelope at every speed.
        r.joint_speed_margin = RewTerm(
            func=joint_speed_margin, weight=w["joint_speed_margin"], params={"threshold": 0.9}
        )
        r.torque_clipping = RewTerm(
            func=torque_clipping, weight=w["torque_clipping"], params={"tolerance": 3.0}
        )
        # Falling (not timing out) always costs more than any penalty saved by ending the episode early.
        r.termination = RewTerm(func=lab_mdp.is_terminated, weight=w["termination"])
        self.scene.robot = self.scene.robot.replace(actuators=spec_actuators())
        add_domain_randomization(self)


def _leg_motor(joints, kp, kd, spec):
    return UnitreeActuatorCfg(
        joint_names_expr=joints,
        stiffness=kp,
        damping=kd,
        armature=0.01,
        effort_limit=spec["y2"],
        effort_limit_sim=spec["y2"],
        velocity_limit=spec["x2"],
        velocity_limit_sim=spec["x2"],
        Y1=spec["y1"],
        Y2=spec["y2"],
        X1=spec["x1"],
        X2=spec["x2"],
        Fs=spec["fs"],
        Fd=spec["fd"],
        min_delay=0,  # action latency randomised per env: 0-4 physics steps (0-20 ms)
        max_delay=4,
    )


def spec_actuators():
    """Actuators bounded by g1lab.actuator_spec (never looser than any source); PD gains of the v0 policy.

    N7520 leg/waist-yaw motors use Unitree's torque-speed curve, motor friction and random latency (explicit
    PD at the 200 Hz physics rate). Other joints stay implicit with the tightest flat torque limit and their
    rated velocity limit.
    """
    hip = SPEC["left_hip_pitch_joint"]
    actuators = {
        "N7520-14.3-hip": _leg_motor([".*_hip_pitch_joint", ".*_hip_yaw_joint"], 100.0, 2.0, hip),
        "N7520-14.3-waist": _leg_motor(["waist_yaw_joint"], 200.0, 5.0, SPEC["waist_yaw_joint"]),
        "N7520-22.5-hip-roll": _leg_motor([".*_hip_roll_joint"], 100.0, 2.0, SPEC["left_hip_roll_joint"]),
        "N7520-22.5-knee": _leg_motor([".*_knee_joint"], 150.0, 4.0, SPEC["left_knee_joint"]),
    }
    flat = {
        "N5020-ankle": ([".*_ankle_.*"], 40.0, 2.0, "left_ankle_pitch_joint"),
        "N5020-waist": (["waist_roll_joint", "waist_pitch_joint"], 200.0, 5.0, "waist_roll_joint"),
        "N5020-arm": (
            [".*_shoulder_.*", ".*_elbow_joint", ".*_wrist_roll_joint"],
            40.0,
            10.0,
            "left_shoulder_pitch_joint",
        ),
        "W4010-wrist": ([".*_wrist_pitch_joint", ".*_wrist_yaw_joint"], 40.0, 10.0, "left_wrist_pitch_joint"),
    }
    for name, (joints, kp, kd, ref) in flat.items():
        spec = SPEC[ref]
        actuators[name] = ImplicitActuatorCfg(
            joint_names_expr=joints,
            stiffness=kp,
            damping=kd,
            armature=0.01,
            effort_limit_sim=spec["y1"],
            velocity_limit_sim=spec["vel"],
        )
    return actuators


def add_domain_randomization(cfg):
    """Physical and actuator randomisation on top of the vendor's (friction 0.3-1.0, torso +[-1, 3] kg, pushes)."""
    e = cfg.events
    e.scale_link_mass = EventTerm(
        func=lab_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    e.torso_com = EventTerm(
        func=lab_mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.03, 0.03), "y": (-0.02, 0.02), "z": (-0.03, 0.03)},
        },
    )
    e.actuator_gains = EventTerm(
        func=lab_mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    e.joint_armature = EventTerm(
        func=lab_mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "armature_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    e.push_robot.interval_range_s = (3.0, 7.0)
    e.push_robot.params["velocity_range"] = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-0.3, 0.3)}


DR_EVENT_NAMES = ("scale_link_mass", "torso_com", "actuator_gains", "joint_armature")
