"""Deterministic Isaac Lab (PhysX) rollout of a 29-DoF G1 actor from ONNX weights or an RSL-RL checkpoint.

Fixed forward command, flat plane, no pushes, no observation noise, no mass/friction randomization.
Only the initial joint velocity noise of the task reset varies between environments (seeded).
PhysX results are not MuJoCo benchmark results; use benchmark_lab_policy.py for the acceptance test.
"""

import argparse
import json
import math
import time
from pathlib import Path

from g1lab.gpu import KIT_ARGS, pin_cuda

pin_cuda()  # must precede any torch/Isaac import

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Unitree-G1-29dof-Velocity-G1LabV0Gains")
parser.add_argument("--policy", required=True, help="policy.onnx or RSL-RL model_N.pt")
parser.add_argument("--usd", required=True)
parser.add_argument("--speed", type=float, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--duration", type=float, default=20.0)
parser.add_argument("--settle", type=float, default=5.0, help="seconds excluded from steady metrics")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--action_std", type=float, default=0.0, help="Gaussian noise added to actions (PPO sampling)"
)
parser.add_argument("--output", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.kit_args = f"{args.kit_args or ''} {KIT_ARGS}".strip()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import unitree_rl_lab.tasks  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from g1lab.isaac_tasks import register  # noqa: E402


def actor_weights(path):
    path = Path(path)
    if path.suffix == ".onnx":
        import onnx
        from onnx import numpy_helper

        graph = onnx.load(str(path)).graph
        return {t.name: torch.from_numpy(numpy_helper.to_array(t).copy()) for t in graph.initializer}
    state = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
    return {k: v for k, v in state.items() if k.startswith("actor.")}


def build_actor(weights, device):
    layers = []
    indices = sorted({int(k.split(".")[1]) for k in weights})
    for n, i in enumerate(indices):
        w, b = weights[f"actor.{i}.weight"], weights[f"actor.{i}.bias"]
        linear = torch.nn.Linear(w.shape[1], w.shape[0])
        linear.weight.data.copy_(w)
        linear.bias.data.copy_(b)
        layers.append(linear)
        if n < len(indices) - 1:
            layers.append(torch.nn.ELU())
    return torch.nn.Sequential(*layers).to(device).eval()


def main():
    register()
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    cfg.scene.robot.spawn.usd_path = str(Path(args.usd).resolve())
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.height_scanner = None
    cfg.curriculum.terrain_levels = None
    cfg.curriculum.lin_vel_cmd_levels = None
    command = cfg.commands.base_velocity
    for ranges in (command.ranges, command.limit_ranges):
        ranges.lin_vel_x = (args.speed, args.speed)
        ranges.lin_vel_y = (0.0, 0.0)
        ranges.ang_vel_z = (0.0, 0.0)
    command.rel_standing_envs = 0.0
    command.resampling_time_range = (1.0e6, 1.0e6)
    command.debug_vis = False
    cfg.observations.policy.enable_corruption = False
    cfg.events.push_robot = None
    cfg.events.add_base_mass = None
    cfg.events.physics_material = None
    cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    cfg.episode_length_s = args.duration + 5.0

    env = gym.make(args.task, cfg=cfg)
    base = env.unwrapped
    robot = base.scene["robot"]
    actor = build_actor(actor_weights(args.policy), base.device)
    obs, _ = env.reset()
    steps = int(round(args.duration / base.step_dt))
    settle = int(round(args.settle / base.step_dt))
    n = args.num_envs
    alive = torch.ones(n, dtype=torch.bool, device=base.device)
    fall_step = torch.full((n,), -1, device=base.device)
    vx_sum = torch.zeros(n, device=base.device)
    vy_sum = torch.zeros(n, device=base.device)
    wz_sq = torch.zeros(n, device=base.device)
    count = torch.zeros(n, device=base.device)
    heading_max = torch.zeros(n, device=base.device)
    started = time.time()
    with torch.inference_mode():
        for step in range(steps):
            action = actor(obs["policy"])
            if args.action_std > 0:
                action = action + args.action_std * torch.randn_like(action)
            obs, _, terminated, truncated, _ = env.step(action)
            fell = (terminated | truncated) & alive
            fall_step[fell] = step
            alive &= ~fell
            q = robot.data.root_quat_w
            yaw = torch.atan2(
                2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2)
            )
            heading_max = torch.where(alive, torch.maximum(heading_max, yaw.abs()), heading_max)
            if step >= settle:
                window = alive.float()
                vx_sum += robot.data.root_lin_vel_b[:, 0] * window
                vy_sum += robot.data.root_lin_vel_b[:, 1] * window
                wz_sq += robot.data.root_ang_vel_b[:, 2] ** 2 * window
                count += window
    survived = alive.cpu()
    denom = count.clamp(min=1)
    per_env = {
        "survived": survived.tolist(),
        "fall_time_s": (fall_step.float() * base.step_dt).cpu().tolist(),
        "steady_vx_mps": (vx_sum / denom).cpu().tolist(),
        "steady_vy_mps": (vy_sum / denom).cpu().tolist(),
        "yaw_rate_rms_rad_s": torch.sqrt(wz_sq / denom).cpu().tolist(),
        "max_abs_heading_deg": torch.rad2deg(heading_max).cpu().tolist(),
    }
    kept = [i for i, s in enumerate(per_env["survived"]) if s]

    def mean(key):
        return sum(per_env[key][i] for i in kept) / len(kept) if kept else math.nan

    summary = {
        "task": args.task,
        "policy": str(Path(args.policy).resolve()),
        "speed_cmd_mps": args.speed,
        "num_envs": n,
        "duration_s": args.duration,
        "settle_s": args.settle,
        "seed": args.seed,
        "action_std": args.action_std,
        "physics": "Isaac Sim PhysX (not MuJoCo)",
        "survival_rate": len(kept) / n,
        "mean_steady_vx_survivors": mean("steady_vx_mps"),
        "mean_yaw_rate_rms_survivors": mean("yaw_rate_rms_rad_s"),
        "max_heading_deg_survivors": max((per_env["max_abs_heading_deg"][i] for i in kept), default=math.nan),
        "wall_time_s": time.time() - started,
        "per_env": per_env,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_env"}, indent=2))
    env.close()


if __name__ == "__main__":
    main()
    app.close()
