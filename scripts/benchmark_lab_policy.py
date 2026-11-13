"""Speed-and-stop audit of the original Unitree RL Lab full-body ONNX policy."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from g1lab.fast_plan import LAB_REVISION, MUJOCO_REVISION
from g1lab.lab_policy import LabReference


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def trial(env, speed, seed, hold=30, video=None, heading_hold=False):
    env.reset(seed)
    rows = []
    renderer = mujoco.Renderer(env.model, height=480, width=640) if video else None
    writer = imageio.get_writer(video, fps=50) if video else None
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17) if video else None
    duration = 5 + hold + 3 + 5
    try:
        for _ in range(round(duration / 0.02)):
            t = env.data.time
            command = speed * np.clip((t - 2) / 3, 0, 1) * np.clip((5 + hold + 3 - t) / 3, 0, 1)
            rotation = env.data.xmat[env.pelvis].reshape(3, 3)
            heading = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
            yaw_command = float(np.clip(-2 * heading, -0.2, 0.2)) if heading_hold else 0.0
            row = env.step([command, 0, yaw_command])
            row["command_yaw_rad_s"] = yaw_command
            rot = env.data.xmat[env.pelvis].reshape(3, 3)
            row["heading_rad"] = float(np.arctan2(rot[1, 0], rot[0, 0]))
            row["max_abs_joint_speed_rad_s"] = float(np.max(np.abs(env.data.qvel[env.vadr])))
            row["mean_abs_action"] = float(np.mean(np.abs(env.action)))
            rows.append(row)
            if renderer:
                camera = mujoco.MjvCamera()
                camera.lookat[:] = env.data.qpos[:3]
                camera.distance, camera.azimuth, camera.elevation = 3.5, 120, -15
                renderer.update_scene(env.data, camera)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 640, 82), fill=(10, 15, 24))
                draw.text((10, 6), "UNITREE RL LAB - public 29-joint ONNX policy", font=font, fill="white")
                draw.text(
                    (10, 31),
                    f"t={row['time_s']:.1f}s | command={command:.2f} | vx={row['vx_mps']:.2f} m/s",
                    font=font,
                    fill="#8bdfff",
                )
                draw.text(
                    (10, 56), "Existing policy baseline - no new RL training", font=font, fill="#ffcf80"
                )
                writer.append_data(np.array(frame))
            if row["fallen"]:
                break
    finally:
        if writer:
            writer.close()
        if renderer:
            renderer.close()
    steady = [r for r in rows if 10 <= r["time_s"] < 5 + hold]
    stopped = [r for r in rows if r["time_s"] >= duration - 2]
    survived = len(rows) == round(duration / 0.02) and not rows[-1]["fallen"]
    vx = float(np.mean([r["vx_mps"] for r in steady])) if steady else None
    rmse = (
        float(np.sqrt(np.mean([(r["vx_mps"] - speed) ** 2 + r["vy_mps"] ** 2 for r in steady])))
        if steady
        else None
    )
    stop = float(np.mean([np.hypot(r["vx_mps"], r["vy_mps"]) for r in stopped])) if stopped else None
    yaw = float(np.sqrt(np.mean([r["yaw_rate"] ** 2 for r in steady]))) if steady else None
    heading = max(abs(r["heading_rad"]) for r in rows)
    passed = bool(
        survived
        and vx is not None
        and vx >= 0.85 * speed
        and rmse <= max(0.15, 0.2 * speed)
        and stop is not None
        and stop < 0.15
        and yaw < 0.25
        and heading < np.pi / 6
    )
    return {
        "command_mps": speed,
        "seed": seed,
        "hold_s": hold,
        "duration_s": len(rows) * 0.02,
        "survived": survived,
        "passed": passed,
        "steady_vx_mps": vx,
        "planar_rmse_mps": rmse,
        "stop_speed_mps": stop,
        "yaw_rate_rmse_rad_s": yaw,
        "max_abs_heading_rad": heading,
    }, rows


def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    env = LabReference(args.source, args.mujoco_source, args.policy_dir)
    protocol = {
        "args": vars(args),
        "lab_commit": LAB_REVISION,
        "mujoco_commit": MUJOCO_REVISION,
        "policy_sha256": hashlib.sha256(env.policy_path.read_bytes()).hexdigest(),
        "policy_contract": "29 actions, 480 observations, term-major history length 5; deployment PD gains",
        "schedule": "stand 2 s, ramp 3 s, hold, ramp down 3 s, stop 5 s; steady metrics after 10 s",
        "gate": "full survival, vx >=85% command, planar RMSE <=max(0.15,20% command), stop speed <0.15 m/s, yaw RMSE <0.25 rad/s, heading <30 degrees",
        "purpose": "Public baseline only; no training; no stage promotion based on target speed alone",
        "heading_hold": "yaw command = clip(-2 * heading, -0.2, 0.2)" if args.heading_hold else "disabled",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    results = []

    def evaluate(speed, seeds, hold, stage):
        for seed in seeds:
            row, trace = trial(env, speed, seed, hold, heading_hold=args.heading_hold)
            row["stage"] = stage
            results.append(row)
            write_csv(output / f"{stage}-{speed:g}-{seed}.csv", trace)
        write_csv(output / "episodes.csv", results)
        group = [r for r in results if r["stage"] == stage and r["command_mps"] == speed]
        print(
            json.dumps(
                {
                    "stage": stage,
                    "command": speed,
                    "passed": sum(r["passed"] for r in group),
                    "survived": sum(r["survived"] for r in group),
                    "count": len(group),
                    "mean_vx": [r["steady_vx_mps"] for r in group],
                }
            ),
            flush=True,
        )
        return all(r["passed"] for r in group)

    good = []
    for speed in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        if evaluate(speed, range(90000, 90003), 30, "coarse"):
            good.append(speed)
    best = max(good) if good else None
    if best is not None:
        for speed in [best + 0.25]:
            if speed <= 3.0 and evaluate(speed, range(90100, 90103), 30, "fine"):
                good.append(speed)
    validated = None
    for speed in sorted(good, reverse=True):
        if evaluate(speed, range(91000, 91005), 60, "validation"):
            validated = speed
            break
    summary = {
        "highest_tested_validated_command_mps": validated,
        "validation": [r for r in results if r["stage"] == "validation"],
        "note": "No new RL training. Finite command/seed audit of an exported baseline, not a global maximum.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    if args.video:
        speed = validated if validated is not None else 0.5
        row, trace = trial(
            env, speed, 91000, 30, str(output / "lab-policy.mp4"), heading_hold=args.heading_hold
        )
        write_csv(output / "video-trajectory.csv", trace)
        (output / "video-episode.json").write_text(json.dumps(row, indent=2))
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="vendor/unitree_rl_lab")
    p.add_argument("--mujoco-source", default="vendor/unitree_mujoco")
    p.add_argument("--output", required=True)
    p.add_argument("--video", action="store_true")
    p.add_argument("--heading-hold", action="store_true")
    p.add_argument(
        "--policy-dir", help="Export directory containing params/deploy.yaml and exported/policy.onnx"
    )
    main(p.parse_args())
