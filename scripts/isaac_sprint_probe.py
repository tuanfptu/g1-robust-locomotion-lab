"""Probe the sprint task in Isaac (PhysX): sign conventions, raw sprint-term magnitudes and reward-term means.

Runs a policy with a fixed forward target (the sprint command ramps toward it), no pushes or noise.
Raw quantities are computed with the speed gate forced open, so they are meaningful at walking speeds too.
"""

import argparse
import json
from pathlib import Path

from g1lab.gpu import KIT_ARGS, pin_cuda

pin_cuda()

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--policy", required=True)
parser.add_argument("--usd", required=True)
parser.add_argument("--speed", type=float, required=True)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--duration", type=float, default=10.0)
parser.add_argument("--output", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.kit_args = f"{args.kit_args or ''} {KIT_ARGS}".strip()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import unitree_rl_lab.tasks  # noqa: E402, F401
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from g1lab import isaac_sprint as sp  # noqa: E402
from g1lab.isaac_tasks import SPRINT_TASK, register  # noqa: E402


def actor_from(path, device):
    path = Path(path)
    if path.suffix == ".onnx":
        import onnx
        from onnx import numpy_helper

        weights = {
            t.name: torch.from_numpy(numpy_helper.to_array(t).copy())
            for t in onnx.load(str(path)).graph.initializer
        }
    else:
        state = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
        weights = {k: v for k, v in state.items() if k.startswith("actor.")}
    layers = []
    idx = sorted({int(k.split(".")[1]) for k in weights})
    for n, i in enumerate(idx):
        lin = torch.nn.Linear(weights[f"actor.{i}.weight"].shape[1], weights[f"actor.{i}.weight"].shape[0])
        lin.weight.data.copy_(weights[f"actor.{i}.weight"])
        lin.bias.data.copy_(weights[f"actor.{i}.bias"])
        layers.append(lin)
        if n < len(idx) - 1:
            layers.append(torch.nn.ELU())
    return torch.nn.Sequential(*layers).to(device).eval()


def main():
    register()
    cfg = parse_env_cfg(SPRINT_TASK, device=args.device, num_envs=args.num_envs)
    cfg.scene.robot.spawn.usd_path = str(Path(args.usd).resolve())
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.height_scanner = None
    cfg.curriculum.terrain_levels = None
    cfg.curriculum.lin_vel_cmd_levels = None
    c = cfg.commands.base_velocity
    c.ranges.lin_vel_x = (args.speed, args.speed)
    c.ranges.ang_vel_z = (0.0, 0.0)
    c.rel_standing_envs = 0.0
    c.resampling_time_range = (1e6, 1e6)
    c.debug_vis = False
    cfg.observations.policy.enable_corruption = False
    cfg.events.push_robot = None
    cfg.events.add_base_mass = None
    cfg.events.physics_material = None
    cfg.episode_length_s = args.duration + 5
    env = gym.make(SPRINT_TASK, cfg=cfg)
    base = env.unwrapped
    actor = actor_from(args.policy, base.device)
    feet_s = SceneEntityCfg(sp.FEET_SENSOR, body_names=sp.FEET, preserve_order=True)
    feet_b = SceneEntityCfg("robot", body_names=sp.FEET, preserve_order=True)
    knees = SceneEntityCfg("robot", body_names=sp.KNEES, preserve_order=True)
    torso = SceneEntityCfg("robot", body_names=["torso_link"])
    ankles = SceneEntityCfg(
        "robot", joint_names=["left_ankle_pitch_joint", "right_ankle_pitch_joint"], preserve_order=True
    )
    shoulders = SceneEntityCfg(
        "robot", joint_names=["left_shoulder_pitch_joint", "right_shoulder_pitch_joint"], preserve_order=True
    )
    hips = SceneEntityCfg(
        "robot", joint_names=["left_hip_pitch_joint", "right_hip_pitch_joint"], preserve_order=True
    )
    for e in (feet_s, feet_b, knees, torso, ankles, shoulders, hips):
        e.resolve(base.scene)
    robot = base.scene["robot"]
    sensor = base.scene.sensors[sp.FEET_SENSOR]
    obs, _ = env.reset()
    rm = base.reward_manager
    names = rm.active_terms
    sums = torch.zeros(len(names), device=base.device)
    stats = {
        k: [] for k in ["toe_minus_heel_at_touchdown_m", "air_time_at_touchdown_s", "stance_at_liftoff_s"]
    }
    acc = {
        "flight": 0.0,
        "double": 0.0,
        "push_bw": 0.0,
        "pitch_deg": 0.0,
        "knee_drop_swing_m": 0.0,
        "n": 0,
        "ns": 0,
    }
    standing = {
        "knee_drop_m": (robot.data.root_pos_w[:, 2:3] - robot.data.body_pos_w[:, knees.body_ids, 2])
        .mean()
        .item(),
        "toe_minus_heel_m": (
            sp._foot_points(base, feet_b, sp.TOE_LOCAL)[..., 2]
            - sp._foot_points(base, feet_b, sp.HEEL_LOCAL)[..., 2]
        )
        .mean()
        .item(),
        "torso_pitch_deg": torch.rad2deg(sp.torso_pitch(base, torso)[0]).mean().item(),
        "default_ankle_pitch": robot.data.default_joint_pos[0, ankles.joint_ids].tolist(),
    }
    alive = torch.ones(args.num_envs, dtype=torch.bool, device=base.device)
    steps = int(args.duration / base.step_dt)
    with torch.inference_mode():
        for step in range(steps):
            obs, rew, term, trunc, _ = env.step(actor(obs["policy"]))
            alive &= ~(term | trunc)
            sums += rm._step_reward[alive].sum(dim=0) if alive.any() else 0
            if step * base.step_dt < 4.0 or not alive.any():
                continue
            g = sp.gait(base, feet_s)
            contact, td, lo = g.contact, g.touchdown, g.liftoff
            diff = -g.landing_drop  # toe minus heel height just before touchdown
            sel = td & alive.unsqueeze(1)
            stats["toe_minus_heel_at_touchdown_m"] += diff[sel].tolist()
            stats["air_time_at_touchdown_s"] += g.last_air[sel].tolist()
            stats["stance_at_liftoff_s"] += g.last_stance[lo & alive.unsqueeze(1)].tolist()
            a = alive
            acc["flight"] += (~contact).all(dim=1)[a].float().sum().item()
            acc["double"] += contact.all(dim=1)[a].float().sum().item()
            acc["push_bw"] += (
                sp.push_off_force(base, "base_velocity", feet_s, SceneEntityCfg("robot"), -2.0, -1.0)[a]
                .sum()
                .item()
            )
            acc["pitch_deg"] += torch.rad2deg(sp.torso_pitch(base, torso)[0])[a].sum().item()
            drop = robot.data.root_pos_w[:, 2:3] - robot.data.body_pos_w[:, knees.body_ids, 2]
            swing = ~contact & a.unsqueeze(1)
            acc["knee_drop_swing_m"] += drop[swing].sum().item()
            acc["ns"] += swing.sum().item()
            acc["n"] += a.sum().item()

    def mean(x):
        return sum(x) / len(x) if x else None

    n = max(acc["n"], 1)

    def median(x):
        return sorted(x)[len(x) // 2] if x else None

    out = {
        "feet_sensor_bodies": sensor.body_names,
        "median_air_time_s": median(stats["air_time_at_touchdown_s"]),
        "median_stance_s": median(stats["stance_at_liftoff_s"]),
        "policy": args.policy,
        "speed_target": args.speed,
        "survival": alive.float().mean().item(),
        "standing": standing,
        "flight_fraction": acc["flight"] / n,
        "double_support_fraction": acc["double"] / n,
        "mean_forward_push_bodyweights": acc["push_bw"] / n,
        "mean_torso_pitch_deg": acc["pitch_deg"] / n,
        "mean_knee_drop_in_swing_m": acc["knee_drop_swing_m"] / max(acc["ns"], 1),
        "toe_minus_heel_at_touchdown_m": mean(stats["toe_minus_heel_at_touchdown_m"]),
        "touchdowns": len(stats["toe_minus_heel_at_touchdown_m"]),
        "mean_air_time_s": mean(stats["air_time_at_touchdown_s"]),
        "mean_stance_s": mean(stats["stance_at_liftoff_s"]),
        "reward_term_mean_per_step": {n_: (sums[i] / max(acc["n"], 1)).item() for i, n_ in enumerate(names)},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "reward_term_mean_per_step"}, indent=2))
    env.close()


if __name__ == "__main__":
    main()
    app.close()
