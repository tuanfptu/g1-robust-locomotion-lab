"""Side-by-side MuJoCo video of two policies under the same command profile (offscreen EGL on the project GPU).

Each panel runs its own MuJoCo world with the shared actuator envelope (spec=True) and heading hold, and follows its
robot with a side camera. Overlay: commanded and measured forward speed, flight phase (both feet off the ground).
Run with .venv-eval.
"""

import argparse
import json
from pathlib import Path

from g1lab.gpu import pin_egl

pin_egl()  # must precede mujoco import; refuses any GPU other than the project GPU

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from g1lab.lab_policy import LabReference  # noqa: E402

W, H = 640, 480


def command_at(t, speed, ramp, hold):
    hold_start = 2.0 + ramp
    return speed * np.clip((t - 2.0) / ramp, 0, 1) * np.clip((hold_start + hold + ramp - t) / ramp, 0, 1)


def main(args):
    panels = [(args.left_label, args.left_policy or None), (args.right_label, args.right_policy)]
    envs = []
    for _, policy in panels:
        env = LabReference(policy_dir=policy, spec=True)
        env.reset(args.seed)
        envs.append(env)
    ramp = max(3.0, args.speed / 2.0)
    duration = 2.0 + ramp + args.hold + ramp + 3.0
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    renderers = [mujoco.Renderer(env.model, height=H, width=W) for env in envs]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(output, fps=50, macro_block_size=8)
    fallen = [False, False]
    stats = [{"vx": [], "flight": 0, "steps": 0, "fell_at": None} for _ in envs]
    try:
        for step in range(int(round(duration / 0.02))):
            frames = []
            for i, env in enumerate(envs):
                t = env.data.time
                cmd = command_at(t, args.speed, ramp, args.hold)
                rot = env.data.xmat[env.pelvis].reshape(3, 3)
                heading = float(np.arctan2(rot[1, 0], rot[0, 0]))
                yaw_cmd = float(np.clip(-1.0 * heading, -0.2, 0.2))
                if not fallen[i]:
                    row = env.step([cmd, 0, yaw_cmd], diagnostics=True)
                    fallen[i] = row["fallen"]
                    if fallen[i]:
                        stats[i]["fell_at"] = round(row["time_s"], 2)
                    flight = not row["contact_l"] and not row["contact_r"]
                    vx = row["vx_mps"]
                    if 2.0 + ramp + 3.0 <= t < 2.0 + ramp + args.hold:
                        stats[i]["vx"].append(vx)
                        stats[i]["flight"] += flight
                        stats[i]["steps"] += 1
                else:
                    env.data.ctrl[:] = 0.0
                    for _ in range(10):  # same 0.02 s per frame as a control step, keeps panels in sync
                        mujoco.mj_step(env.model, env.data)
                    flight, vx = False, 0.0
                camera = mujoco.MjvCamera()
                camera.lookat[:] = env.data.qpos[:3]
                camera.distance, camera.azimuth, camera.elevation = 3.2, 90, -12
                renderers[i].update_scene(env.data, camera)
                img = Image.fromarray(renderers[i].render())
                draw = ImageDraw.Draw(img)
                draw.rectangle((0, 0, W, 78), fill=(10, 15, 24))
                draw.text((10, 6), panels[i][0], font=font, fill="white")
                draw.text(
                    (10, 32),
                    f"t={env.data.time:5.1f}s  lenh={cmd:4.2f}  thuc={vx:4.2f} m/s",
                    font=small,
                    fill="#8bdfff",
                )
                status = "NGA" if fallen[i] else ("PHA BAY" if flight else "cham dat")
                color = "#ff6b6b" if fallen[i] else ("#7dff8a" if flight else "#ffcf80")
                draw.text((10, 54), status, font=small, fill=color)
                frames.append(np.array(img))
            writer.append_data(np.concatenate(frames, axis=1))
    finally:
        writer.close()
        for r in renderers:
            r.close()
    summary = {
        "output": str(output.resolve()),
        "speed_cmd_mps": args.speed,
        "seed": args.seed,
        "panels": [
            {
                "label": panels[i][0],
                "policy": panels[i][1] or "vendor ONNX v0",
                "mean_vx_hold_mps": float(np.mean(s["vx"])) if s["vx"] else None,
                "flight_fraction_hold": s["flight"] / s["steps"] if s["steps"] else None,
                "fell_at_s": s["fell_at"],
            }
            for i, s in enumerate(stats)
        ],
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-policy", default="", help="policy dir; empty = public ONNX v0 (baseline)")
    parser.add_argument("--left-label", default="Baseline: ONNX goc Unitree")
    parser.add_argument("--right-policy", required=True)
    parser.add_argument("--right-label", default="Moi: sprint v2")
    parser.add_argument("--speed", type=float, required=True)
    parser.add_argument("--hold", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=99100)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
