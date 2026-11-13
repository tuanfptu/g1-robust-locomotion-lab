"""Inspectable Isaac Lab speed-stage commands. No simulator imports required."""

import json
import re
from pathlib import Path

from g1lab.gpu import KIT_ARGS
from g1lab.isaac_tasks import SPRINT_TASK, V0_GAINS_TASK

LAB_REVISION = "4960b84732b0c2ec593dccbfe963fda1bcd7b1e3"
MUJOCO_REVISION = "1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d"
# The pinned Unitree train.py ignores --experiment_name; the Hydra agent override is what takes effect.
EXPERIMENT_NAME = "g1_fast"
SPRINT_EXPERIMENT = "g1_sprint"


def stage_settings(cap):
    if cap not in (1.0, 1.5, 2.0, 2.5, 3.0):
        raise ValueError("Choose a staged cap: 1, 1.5, 2, 2.5 or 3 m/s")
    period = {1.0: 0.8, 1.5: 0.65, 2.0: 0.55, 2.5: 0.5, 3.0: 0.45}[cap]
    return {"cap_mps": cap, "gait_period_s": period, "stance_fraction": 0.55 if cap == 1 else 0.45}


WRAPPER = Path(__file__).resolve().parents[2] / "scripts/isaac_train.py"


def command(
    source,
    python,
    usd,
    cap,
    envs=512,
    iterations=1000,
    seed=42,
    checkpoint=None,
    task=V0_GAINS_TASK,
    critic_warmup=0,
):
    settings = stage_settings(cap)
    if envs < 1 or iterations < 1:
        raise ValueError("Environment count and iterations must be positive")
    if critic_warmup and not checkpoint:
        raise ValueError("Critic warm-up only makes sense when resuming a transplanted actor")
    if critic_warmup >= iterations:
        raise ValueError("Critic warm-up must leave at least one actor-update iteration")
    usd = Path(usd).resolve()
    argv = [
        str(python),
        str(WRAPPER),
        str(Path(source).resolve() / "scripts/rsl_rl/train.py"),
        "--task",
        task,
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
        EXPERIMENT_NAME,
        "--run_name",
        f"cap_{cap:g}_seed_{seed}",
        "--logger",
        "tensorboard",
        "--kit_args",
        KIT_ARGS,
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
    if critic_warmup:
        argv += ["--critic_warmup", str(critic_warmup)]
    overrides = {
        "agent.experiment_name": EXPERIMENT_NAME,
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


def sprint_command(
    source,
    python,
    usd,
    envs=4096,
    iterations=1000,
    seed=42,
    checkpoint=None,
    critic_warmup=0,
    extra=(),
    learning_rate=None,
):
    """Sprint-task training command (flight-phase running). Gait/curriculum come from the task config."""
    if envs < 1 or iterations < 1:
        raise ValueError("Environment count and iterations must be positive")
    if critic_warmup and not checkpoint:
        raise ValueError("Critic warm-up only makes sense when resuming a transplanted actor")
    if critic_warmup >= iterations:
        raise ValueError("Critic warm-up must leave at least one actor-update iteration")
    argv = [
        str(python),
        str(WRAPPER),
        str(Path(source).resolve() / "scripts/rsl_rl/train.py"),
        "--task",
        SPRINT_TASK,
        "--headless",
        "--device",
        "cuda:0",
        "--num_envs",
        str(envs),
        "--max_iterations",
        str(iterations),
        "--seed",
        str(seed),
        "--run_name",
        f"sprint_seed_{seed}",
        "--logger",
        "tensorboard",
        "--kit_args",
        KIT_ARGS,
        *extra,
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
    if critic_warmup:
        argv += ["--critic_warmup", str(critic_warmup)]
    overrides = {
        "agent.experiment_name": SPRINT_EXPERIMENT,
        "env.scene.robot.spawn.usd_path": json.dumps(str(Path(usd).resolve())),
        "agent.algorithm.learning_rate": str(learning_rate or (1.0e-4 if checkpoint else 1.0e-3)),
    }
    argv.extend(f"{key}={value}" for key, value in overrides.items())
    return argv
