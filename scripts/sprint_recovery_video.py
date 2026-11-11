"""Actual-physics stress-test video using current policies; not a solved sprint demo."""

import argparse
import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from g1lab.config import EnvConfig
from g1lab.recovery import transfer_state
from g1lab.training import load_policy, policy_action


def main(args):
    torch.set_num_threads(1)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    configuration = EnvConfig(random_commands=False, command=(0, 0, 0), episode_seconds=args.seconds)
    walk, walk_vec, walk_env = load_policy(args.walk, configuration)
    recovery, recovery_vec, recovery_env = load_policy(args.recovery, replace(configuration, task="recovery"))
    if walk_env.config.task != "locomotion" or recovery_env.config.task != "recovery":
        raise ValueError("Provide locomotion and recovery policies in that order")
    for env in (walk_env, recovery_env):
        env.render_mode = "rgb_array"
    manifest = {
        "args": vars(args),
        "purpose": "Current-policy stress test, not trained sprint/recovery",
        "environment_walk": walk_env.config.to_dict(),
        "environment_recovery": recovery_env.config.to_dict(),
        "handoff": "Copies qpos/qvel/time, changes to target policy PD/observation/action contract; unvalidated transition",
        "policy_sha256": {
            bundle: hashlib.sha256((Path(bundle) / "policy.zip").read_bytes()).hexdigest()
            for bundle in (args.walk, args.recovery)
        },
    }
    (output / "protocol.json").write_text(json.dumps(manifest, indent=2))
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    rows, events = [], []
    mode, recovery_time, max_speed = "LOCOMOTION", None, 0.0
    active, model, vec = walk_env, walk, walk_vec
    walk_env.reset(seed=args.seed)
    recovery_env.reset(seed=args.seed)
    pushed = False
    try:
        with imageio.get_writer(output / "sprint-fall-recovery.mp4", fps=50) as writer:
            for step in range(round(args.seconds / 0.02)):
                time = step * 0.02
                target_speed = 0 if time < 2 else (0.3 if time < 6 else min(2.0, 0.3 + (time - 6) * 0.3))
                if mode == "LOCOMOTION":
                    active.set_command(target_speed, 0, 0)
                    if time >= 14 and not pushed:
                        active.push(40, np.pi / 2)
                        pushed = True
                        events.append({"time_s": time, "event": "40 Ns lateral stress impulse"})
                else:
                    active.set_command(0, 0, 0)
                action = policy_action(model, vec, active._observation())
                _, _, terminated, _, info = active.step(action)
                velocity = np.zeros(6)
                mujoco.mj_objectVelocity(
                    active.model, active.data, mujoco.mjtObj.mjOBJ_BODY, active.pelvis, velocity, 0
                )
                speed = float(np.linalg.norm(velocity[3:5]))
                if mode == "LOCOMOTION" and not info["fallen"]:
                    max_speed = max(max_speed, speed)
                rows.append(
                    {
                        "time_s": (step + 1) * 0.02,
                        "mode": mode,
                        "target_vx_mps": target_speed if mode == "LOCOMOTION" else 0,
                        "horizontal_speed_mps": speed,
                        "pelvis_height_m": info["base_height_m"],
                        "upright_cos": info["upright"],
                        "recovery_success": info.get("recovery_success", False),
                    }
                )
                frame = Image.fromarray(active.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 640, 103), fill=(10, 14, 22))
                draw.text(
                    (12, 7), "G1: sprint / fall / recovery - CURRENT PILOT", font=title_font, fill="white"
                )
                draw.text(
                    (12, 36), f"t={time:.1f}s | {mode} | speed={speed:.2f} m/s", font=font, fill="#74e2ff"
                )
                draw.text(
                    (12, 59),
                    f"Target vx={rows[-1]['target_vx_mps']:.2f} m/s | height={info['base_height_m']:.2f} m",
                    font=font,
                    fill="white",
                )
                draw.text(
                    (12, 81),
                    "Fast running and get-up are NOT solved by these policies.",
                    font=font,
                    fill="#ffd080",
                )
                writer.append_data(np.array(frame))
                if mode == "LOCOMOTION" and info["fallen"]:
                    events.append(
                        {"time_s": (step + 1) * 0.02, "event": "fall detected; handoff to recovery pilot"}
                    )
                    transfer_state(active, recovery_env)
                    active, model, vec = recovery_env, recovery, recovery_vec
                    mode = "RECOVERY ATTEMPT"
                elif mode == "RECOVERY ATTEMPT" and terminated:
                    recovery_time = (step + 1) * 0.02
                    events.append({"time_s": recovery_time, "event": "recovery success; stop command"})
                    transfer_state(active, walk_env)
                    active, model, vec = walk_env, walk, walk_vec
                    mode = "STABILIZE"
    finally:
        walk_vec.close()
        recovery_vec.close()
    with (output / "trajectory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "events": events,
        "max_horizontal_speed_before_fall_mps": max_speed,
        "recovered": recovery_time is not None,
        "recovery_success_at_s": recovery_time,
        "final_mode": mode,
        "duration_s": args.seconds,
        "note": "No animation, pose teleportation or speed-up. This is not a validated FSM controller.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walk", default="runs/autonomous-1660/selected-terrain")
    parser.add_argument("--recovery", default="runs/recovery-supine-1660-v1/final")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=float, default=26)
    parser.add_argument("--seed", type=int, default=63000)
    main(parser.parse_args())
