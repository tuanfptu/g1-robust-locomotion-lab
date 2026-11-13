"""Speed-ladder evaluation of a sprint checkpoint in Isaac (PhysX), one process, offline-safe.

Each group of envs gets one target speed; every robot starts from rest, the sprint command ramps toward the
target with the task's acceleration limit, then the speed is held. Steady metrics are taken over
[ramp_end + settle, ramp_end + settle + hold]. Deterministic actor (mean action), no pushes, no observation
noise, nominal friction and mass. PhysX only: the MuJoCo benchmark remains a separate check.
"""

import argparse
import json
import time
from pathlib import Path

from g1lab.gpu import KIT_ARGS, pin_cuda

pin_cuda()

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--policy", required=True, help="RSL-RL model_N.pt or policy.onnx")
parser.add_argument("--usd", required=True)
parser.add_argument("--speeds", default="1,2,3,4,5,6,7,8,9,10,11")
parser.add_argument("--envs_per_speed", type=int, default=16)
parser.add_argument("--settle", type=float, default=2.0)
parser.add_argument("--hold", type=float, default=10.0)
parser.add_argument("--seed", type=int, default=0)
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
    idx = sorted({int(k.split(".")[1]) for k in weights})
    layers = []
    for n, i in enumerate(idx):
        w, b = weights[f"actor.{i}.weight"], weights[f"actor.{i}.bias"]
        lin = torch.nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data.copy_(w)
        lin.bias.data.copy_(b)
        layers.append(lin)
        if n < len(idx) - 1:
            layers.append(torch.nn.ELU())
    return torch.nn.Sequential(*layers).to(device).eval()


def main():
    register()
    speeds = [float(s) for s in args.speeds.split(",")]
    per = args.envs_per_speed
    n = per * len(speeds)
    cfg = parse_env_cfg(SPRINT_TASK, device=args.device, num_envs=n)
    cfg.seed = args.seed
    cfg.scene.robot.spawn.usd_path = str(Path(args.usd).resolve())
    cfg.curriculum.lin_vel_cmd_levels = None
    c = cfg.commands.base_velocity
    c.ranges.lin_vel_x = (0.0, 0.0)
    c.ranges.ang_vel_z = (0.0, 0.0)
    c.limit_ranges.lin_vel_x = (0.0, max(speeds))
    c.rel_standing_envs = 0.0
    c.resampling_time_range = (1.0e6, 1.0e6)
    cfg.observations.policy.enable_corruption = False
    cfg.events.push_robot = None
    cfg.events.add_base_mass = None
    cfg.events.physics_material = None
    accel = c.accel_limit
    duration = max(speeds) / accel + args.settle + args.hold
    cfg.episode_length_s = duration + 5.0
    env = gym.make(SPRINT_TASK, cfg=cfg)
    base = env.unwrapped
    robot = base.scene["robot"]
    term = base.command_manager.get_term("base_velocity")
    actor = actor_from(args.policy, base.device)
    feet = SceneEntityCfg(sp.FEET_SENSOR, body_names=sp.FEET, preserve_order=True)
    feet.resolve(base.scene)
    torso = SceneEntityCfg("robot", body_names=["torso_link"])
    torso.resolve(base.scene)

    target = torch.tensor(speeds, device=base.device).repeat_interleave(per)
    start = target / accel + args.settle
    stop = start + args.hold
    obs, _ = env.reset()
    term.target_b[:, 0] = target
    alive = torch.ones(n, dtype=torch.bool, device=base.device)
    keys = [
        "vx",
        "vy",
        "wz2",
        "flight",
        "double",
        "count",
        "pitch",
        "touch",
        "forefoot",
        "stance",
        "lift",
        "air",
    ]
    acc = {k: torch.zeros(n, device=base.device) for k in keys}
    steps = int(round(duration / base.step_dt))
    started = time.time()
    with torch.inference_mode():
        for step in range(steps):
            term.target_b[:, 0] = torch.where(alive, target, term.target_b[:, 0])
            obs, _, terminated, truncated, _ = env.step(actor(obs["policy"]))
            alive &= ~(terminated | truncated)
            t = (step + 1) * base.step_dt
            window = alive & (t >= start) & (t < stop)
            w = window.float()
            g = sp.gait(base, feet)
            acc["vx"] += robot.data.root_lin_vel_b[:, 0] * w
            acc["vy"] += robot.data.root_lin_vel_b[:, 1] * w
            acc["wz2"] += robot.data.root_ang_vel_b[:, 2] ** 2 * w
            acc["flight"] += (~g.contact).all(dim=1).float() * w
            acc["double"] += g.contact.all(dim=1).float() * w
            acc["pitch"] += torch.rad2deg(sp.torso_pitch(base, torso)[0]) * w
            acc["count"] += w
            touch = g.touchdown.float() * w.unsqueeze(1)
            acc["touch"] += touch.sum(dim=1)
            acc["forefoot"] += ((g.landing_drop > 0.0).float() * touch).sum(dim=1)
            lift = g.liftoff.float() * w.unsqueeze(1)
            acc["lift"] += lift.sum(dim=1)
            acc["stance"] += (g.last_stance * lift).sum(dim=1)
            acc["air"] += (g.last_air * touch).sum(dim=1)
    groups = []
    for i, v in enumerate(speeds):
        sl = slice(i * per, (i + 1) * per)
        ok = alive[sl]
        k = int(ok.sum())

        def m(key, den="count", sl=sl, ok=ok, k=k):
            if k == 0:
                return None
            d = acc[den][sl][ok].clamp(min=1)
            return float((acc[key][sl][ok] / d).mean())

        vx = m("vx")
        groups.append(
            {
                "speed_cmd_mps": v,
                "survival": k / per,
                "mean_vx_mps": vx,
                "vx_ratio": vx / v if vx is not None else None,
                "mean_abs_vy_mps": abs(m("vy")) if k else None,
                "yaw_rate_rms": m("wz2") ** 0.5 if k else None,
                "flight_fraction": m("flight"),
                "double_support_fraction": m("double"),
                "forefoot_touchdown_fraction": m("forefoot", "touch"),
                "mean_stance_s": m("stance", "lift"),
                "mean_air_s": m("air", "touch"),
                "cadence_steps_per_s": (
                    float(
                        (acc["touch"][sl][ok] / (acc["count"][sl][ok] * base.step_dt).clamp(min=1e-6)).mean()
                    )
                    if k
                    else None
                ),
                "mean_torso_pitch_deg": m("pitch"),
            }
        )
    passed = [
        g for g in groups if g["survival"] >= 0.9 and g["vx_ratio"] is not None and g["vx_ratio"] >= 0.85
    ]
    out = {
        "policy": str(Path(args.policy).resolve()),
        "physics": "Isaac Sim PhysX (training simulator)",
        "accel_limit_mps2": accel,
        "settle_s": args.settle,
        "hold_s": args.hold,
        "envs_per_speed": per,
        "seed": args.seed,
        "criteria": "survival >= 0.9 and mean vx >= 0.85 * command over the hold window",
        "max_sustained_speed_mps": max((g["speed_cmd_mps"] for g in passed), default=0.0),
        "max_mean_vx_mps": max((g["mean_vx_mps"] for g in groups if g["survival"] >= 0.9), default=0.0),
        "wall_time_s": time.time() - started,
        "groups": groups,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("max_sustained_speed_mps", "max_mean_vx_mps", "wall_time_s")}))
    env.close()


if __name__ == "__main__":
    main()
    app.close()
