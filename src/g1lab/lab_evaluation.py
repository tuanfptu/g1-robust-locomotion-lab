"""Reusable speed/stop evaluation for the full-body Unitree ONNX policy."""

import csv

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def trial(
    env,
    speed,
    seed,
    hold=30,
    video=None,
    heading_hold=False,
    heading_gain=2.0,
    ramp=3.0,
    diagnostics=False,
):
    if speed <= 0 or hold <= 5 or ramp <= 0 or heading_gain < 0:
        raise ValueError("Require positive speed/ramp, hold >5 s and nonnegative heading gain")
    env.reset(seed)
    rows = []
    renderer = mujoco.Renderer(env.model, height=480, width=640) if video else None
    writer = imageio.get_writer(video, fps=50) if video else None
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17) if video else None
    hold_start = 2 + ramp
    duration = hold_start + hold + ramp + 5
    try:
        for _ in range(round(duration / 0.02)):
            t = env.data.time
            command = (
                speed * np.clip((t - 2) / ramp, 0, 1) * np.clip((hold_start + hold + ramp - t) / ramp, 0, 1)
            )
            rotation = env.data.xmat[env.pelvis].reshape(3, 3)
            heading = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
            yaw_command = float(np.clip(-heading_gain * heading, -0.2, 0.2)) if heading_hold else 0.0
            row = env.step([command, 0, yaw_command], diagnostics=diagnostics)
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
                    (10, 56),
                    f"Heading hold {'ON' if heading_hold else 'OFF'} | no new RL training",
                    font=font,
                    fill="#ffcf80",
                )
                writer.append_data(np.array(frame))
            if row["fallen"]:
                break
    finally:
        if writer:
            writer.close()
        if renderer:
            renderer.close()
    steady = [r for r in rows if hold_start + 5 <= r["time_s"] < hold_start + hold]
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
    result = {
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
    }
    if diagnostics:
        result.update(
            {
                "max_tilt_deg": float(
                    np.degrees(np.arccos(np.clip(min(r["upright_cos"] for r in rows), -1, 1)))
                ),
                "steady_requested_torque_clipped_fraction": float(
                    np.mean([r["requested_torque_clipped_fraction"] for r in steady])
                )
                if steady
                else None,
                "steady_peak_torque_utilization": max(
                    (r["peak_torque_utilization"] for r in steady), default=None
                ),
                "steady_contact_foot_speed_mps": float(
                    np.mean([r["contact_foot_speed_mps"] for r in steady if r["feet_in_contact"] > 0])
                )
                if any(r["feet_in_contact"] > 0 for r in steady)
                else None,
            }
        )
    return result, rows
