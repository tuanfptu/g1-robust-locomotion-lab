"""Interactive native MuJoCo viewer. Keyboard commands are handled by the viewer."""

import time
from collections import deque

import mujoco.viewer
import numpy as np

from .config import EnvConfig
from .env import G1Env
from .training import load_policy, policy_action


def demo(args):
    config = EnvConfig(
        terrain=args.terrain,
        difficulty=args.difficulty,
        random_commands=False,
        command=(0, 0, 0),
        episode_seconds=120,
    )
    teacher = None
    if getattr(args, "teacher", None):
        from .bootstrap import UnitreeTeacher, teacher_config

        teacher = UnitreeTeacher(args.teacher)
        config = teacher_config(config)
        model, vec, env = None, None, G1Env(config)
        print("PRETRAINED UNITREE TEACHER on project G1; not a from-scratch trained policy.")
    elif args.bundle:
        model, vec, env = load_policy(args.bundle, config)
    else:
        model, vec, env = None, None, G1Env(config)
    config = env.config
    events = deque()
    print("W/S: forward/back, A/D: lateral, Q/E: yaw, X: stop, P: push, R: reset. Close window to exit.")
    if model is None and teacher is None:
        print("PD diagnostic mode: no learned walking policy is loaded.")
    obs, _ = env.reset(seed=args.seed)
    automatic = getattr(args, "auto", False)
    if config.task == "recovery":
        automatic = False
        print("Recovery pilot: starts lying down; success requires one second of stable standing.")
    if model is not None and hasattr(model, "bootstrap_origin"):
        print("Student initialized from Unitree teacher; see checkpoint origin.json for provenance.")
    if automatic:
        print("Automatic 40-second cycle: stand → walk → gentle turn → stop. Close window to exit.")
    try:
        with mujoco.viewer.launch_passive(
            env.model, env.data, key_callback=None if automatic else events.append
        ) as viewer:
            viewer.opt.geomgroup[1] = 0
            viewer.opt.geomgroup[3] = 1
            viewer.cam.distance = 3
            while viewer.is_running():
                started = time.perf_counter()
                while events:
                    key = events.popleft()
                    if key in [ord(k) for k in "WSADQEX"]:
                        automatic = False
                    if key in (ord("W"), ord("S")):
                        env.command[0] = np.clip(env.command[0] + (0.1 if key == ord("W") else -0.1), -0.3, 1)
                    elif key in (ord("A"), ord("D")):
                        env.command[1] = np.clip(
                            env.command[1] + (0.1 if key == ord("A") else -0.1), -0.4, 0.4
                        )
                    elif key in (ord("Q"), ord("E")):
                        env.command[2] = np.clip(env.command[2] + (0.1 if key == ord("Q") else -0.1), -1, 1)
                    elif key == ord("X"):
                        env.command[:] = 0
                    elif key == ord("P"):
                        env.push(5, np.pi / 2)
                    elif key == ord("R"):
                        obs, _ = env.reset()
                        if teacher is not None:
                            teacher.reset()
                        if env.has_hfield:
                            viewer.update_hfield(0)
                    print(f"command={env.command.round(2)}")
                if automatic:
                    phase = env.data.time % 40
                    command = (
                        (0, 0, 0)
                        if phase < 3 or phase >= 30
                        else ((0.3, 0, 0) if phase < 20 else (0.2, 0, 0.2))
                    )
                    env.set_command(*command)
                obs = env._observation()
                action = (
                    teacher.action(env)
                    if teacher is not None
                    else (policy_action(model, vec, obs) if model else np.zeros(env.model.nu))
                )
                obs, _, terminated, truncated, info = env.step(action)
                with viewer.lock():
                    viewer.cam.lookat[:] = env.data.qpos[:3]
                viewer.sync()
                if env.step_count % 50 == 0:
                    print(
                        f"t={info['time']:.1f} vx={info['vx']:.2f} vy={info['vy']:.2f} "
                        f"height={info['base_height_m']:.2f}"
                    )
                if terminated or truncated:
                    print("Episode ended; resetting.")
                    obs, _ = env.reset()
                    if teacher is not None:
                        teacher.reset()
                    if env.has_hfield:
                        viewer.update_hfield(0)
                time.sleep(max(0, config.control_dt - (time.perf_counter() - started)))
    finally:
        vec.close() if vec is not None else env.close()
