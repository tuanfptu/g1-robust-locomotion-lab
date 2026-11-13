"""Inspectable Isaac Lab speed-stage commands. No simulator imports required."""

import json
import re
from pathlib import Path

LAB_REVISION = "4960b84732b0c2ec593dccbfe963fda1bcd7b1e3"
MUJOCO_REVISION = "1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d"


def stage_settings(cap):
    if cap not in (1.0, 1.5, 2.0, 2.5, 3.0):
        raise ValueError("Choose a staged cap: 1, 1.5, 2, 2.5 or 3 m/s")
    period = {1.0: 0.8, 1.5: 0.65, 2.0: 0.55, 2.5: 0.5, 3.0: 0.45}[cap]
    return {"cap_mps": cap, "gait_period_s": period, "stance_fraction": 0.55 if cap == 1 else 0.45}


def command(source, python, usd, cap, envs=512, iterations=1000, seed=42, checkpoint=None):
    settings = stage_settings(cap)
    if envs < 1 or iterations < 1:
        raise ValueError("Environment count and iterations must be positive")
    usd = Path(usd).resolve()
    argv = [
        str(python),
        str(Path(source).resolve() / "scripts/rsl_rl/train.py"),
        "--task",
        "Unitree-G1-29dof-Velocity",
        "--headless",
        "--device",
        "cuda:0",
        "--num_envs",
        str(envs),
        "--max_iterations",
        str(iterations),
        "--seed",
        str(seed),
        "--experiment_name",
        "g1_fast",
        "--run_name",
        f"cap_{cap:g}_seed_{seed}",
        "--logger",
        "tensorboard",
    ]
    if checkpoint:
        checkpoint = Path(checkpoint).resolve()
        if checkpoint.suffix != ".pt":
            raise ValueError("Resume needs a full RSL-RL .pt checkpoint; exported ONNX cannot resume PPO")
        argv += [
            "--resume",
            "--load_run",
            re.escape(checkpoint.parent.name),
            "--checkpoint",
            re.escape(checkpoint.name),
        ]
    overrides = {
        "env.scene.robot.spawn.usd_path": json.dumps(str(usd)),
        "env.scene.terrain.terrain_type": "plane",
        "env.scene.terrain.terrain_generator": "null",
        "env.scene.height_scanner": "null",
        "env.curriculum.terrain_levels": "null",
        "env.curriculum.lin_vel_cmd_levels": "null",
        "env.commands.base_velocity.ranges.lin_vel_x": f"[0.0,{cap}]",
        "env.commands.base_velocity.ranges.lin_vel_y": "[0.0,0.0]",
        "env.commands.base_velocity.ranges.ang_vel_z": "[-0.2,0.2]",
        "env.commands.base_velocity.limit_ranges.lin_vel_x": f"[0.0,{cap}]",
        "env.commands.base_velocity.limit_ranges.lin_vel_y": "[0.0,0.0]",
        "env.commands.base_velocity.limit_ranges.ang_vel_z": "[-0.2,0.2]",
        "env.commands.base_velocity.resampling_time_range": "[5.0,8.0]",
        "env.commands.base_velocity.rel_standing_envs": "0.1",
        "env.events.reset_base.params.pose_range.yaw": "[0.0,0.0]",
        "env.terminations.base_height.params.minimum_height": "0.45",
        "env.terminations.bad_orientation.params.limit_angle": "0.7",
        "env.rewards.gait.params.period": str(settings["gait_period_s"]),
        "env.rewards.gait.params.threshold": str(settings["stance_fraction"]),
        "agent.algorithm.learning_rate": "0.0001" if checkpoint else "0.001",
    }
    argv.extend(f"{key}={value}" for key, value in overrides.items())
    return argv
