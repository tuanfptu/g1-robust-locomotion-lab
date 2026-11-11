"""Speed sweep of the original Unitree TorchScript policy on its original 12-DoF model."""

import argparse
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from unitree_reference import DEFAULT, KD, KP

from g1lab.terrain import add_terrain


class Reference:
    def __init__(self, source):
        source = Path(source).resolve()
        root = ET.parse(source / "resources/robots/g1_description/g1_12dof.xml").getroot()
        root.find("compiler").set("meshdir", str(source / "resources/robots/g1_description/meshes"))
        ET.SubElement(root.find("worldbody"), "light", pos="0 0 5", dir="0 0 -1", directional="true")
        add_terrain(root, "flat", 0)
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.model.opt.timestep = 0.002
        self.model.geom_friction[:, 0] = 0.8
        self.data = mujoco.MjData(self.model)
        self.policy = torch.jit.load(str(source / "deploy/pre_train/g1/motion.pt"), map_location="cpu")
        self.policy.eval()
        self.pelvis = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    def trial(self, speed, seed, hold=30, video=None):
        mujoco.mj_resetData(self.model, self.data)
        self.policy.reset_memory()
        self.data.qpos[7:] = DEFAULT + np.random.default_rng(seed).uniform(-0.005, 0.005, 12)
        mujoco.mj_forward(self.model, self.data)
        action = np.zeros(12)
        samples = []
        if video:
            # Plane contacts are infinite; enlarge its visual patch for long-distance footage.
            ground = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
            self.model.geom_size[ground, :2] = 200
        renderer = mujoco.Renderer(self.model, height=480, width=640) if video else None
        writer = imageio.get_writer(video, fps=50) if video else None
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17) if video else None
        fell = False
        try:
            for _ in range(round((5 + hold) / 0.02)):
                t = self.data.time
                command = np.array([speed * np.clip((t - 2) / 3, 0, 1), 0, 0])
                rotation = self.data.xmat[self.pelvis].reshape(3, 3)
                phase = 2 * np.pi * (t % 0.8) / 0.8
                obs = np.concatenate(
                    [
                        self.data.qvel[3:6] * 0.25,
                        rotation.T @ [0, 0, -1],
                        command * [2, 2, 0.25],
                        self.data.qpos[7:] - DEFAULT,
                        self.data.qvel[6:] * 0.05,
                        action,
                        [np.sin(phase), np.cos(phase)],
                    ]
                ).astype(np.float32)
                with torch.inference_mode():
                    action = self.policy(torch.from_numpy(obs[None])).numpy()[0]
                target = DEFAULT + action * 0.25
                for _ in range(10):
                    self.data.ctrl[:] = (target - self.data.qpos[7:]) * KP - self.data.qvel[6:] * KD
                    mujoco.mj_step(self.model, self.data)
                body, world = np.zeros(6), np.zeros(6)
                mujoco.mj_objectVelocity(
                    self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.pelvis, body, 1
                )
                mujoco.mj_objectVelocity(
                    self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.pelvis, world, 0
                )
                upright = float(self.data.xmat[self.pelvis].reshape(3, 3)[2, 2])
                finite = np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
                fell = bool(self.data.qpos[2] < 0.45 or upright < 0.45 or not finite)
                samples.append(
                    {
                        "time_s": float(self.data.time),
                        "command_mps": float(command[0]),
                        "body_vx_mps": float(body[3]),
                        "body_vy_mps": float(body[4]),
                        "world_vx_mps": float(world[3]),
                        "horizontal_speed_mps": float(np.linalg.norm(world[3:5])),
                        "upright_cos": upright,
                        "height_m": float(self.data.qpos[2]),
                        "fallen": fell,
                    }
                )
                if renderer:
                    camera = mujoco.MjvCamera()
                    camera.lookat[:] = self.data.qpos[:3]
                    camera.distance, camera.azimuth, camera.elevation = 3.5, 120, -15
                    renderer.update_scene(self.data, camera)
                    frame = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0, 0, 640, 82), fill=(12, 16, 24))
                    draw.text(
                        (10, 6), "ORIGINAL UNITREE POLICY - original 12-DoF model", font=font, fill="white"
                    )
                    draw.text(
                        (10, 31),
                        f"t={self.data.time:.1f}s | target={command[0]:.2f} m/s | measured vx={body[3]:.2f}",
                        font=font,
                        fill="#83ddff",
                    )
                    draw.text(
                        (10, 56),
                        f"Flat ground | seed={seed} | " + ("FALL" if fell else "real-time 50 fps"),
                        font=font,
                        fill="#ffd080",
                    )
                    writer.append_data(np.array(frame))
                if fell:
                    break
        finally:
            if writer:
                writer.close()
            if renderer:
                renderer.close()
        steady = [r for r in samples if r["time_s"] >= 10]
        mean_vx = float(np.mean([r["body_vx_mps"] for r in steady])) if steady else None
        rmse = (
            float(np.sqrt(np.mean([(r["body_vx_mps"] - speed) ** 2 + r["body_vy_mps"] ** 2 for r in steady])))
            if steady
            else None
        )
        survived = not fell and len(samples) == round((5 + hold) / 0.02)
        passed = bool(
            survived and mean_vx is not None and mean_vx >= 0.8 * speed and rmse <= max(0.2, 0.2 * speed)
        )
        row = {
            "command_mps": speed,
            "seed": seed,
            "hold_s": hold,
            "duration_s": len(samples) * 0.02,
            "survived": survived,
            "passed_tracking": passed,
            "steady_mean_body_vx_mps": mean_vx,
            "steady_rmse_mps": rmse,
            "steady_mean_world_vx_mps": float(np.mean([r["world_vx_mps"] for r in steady]))
            if steady
            else None,
            "peak_horizontal_speed_mps": max(r["horizontal_speed_mps"] for r in samples if not r["fallen"])
            if any(not r["fallen"] for r in samples)
            else 0.0,
        }
        return row, samples


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    torch.set_num_threads(1)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "args": vars(args),
        "policy": "Original Unitree motion.pt, no training or student",
        "sha256": hashlib.sha256(
            (Path(args.source) / "deploy/pre_train/g1/motion.pt").read_bytes()
        ).hexdigest(),
        "source_commit": "276801e46c5d433564f24658bac64f254b7d2d4b",
        "model": "Original Unitree 12 controlled leg DoF, fixed upper body",
        "conditions": "flat, friction 0.8, no pushes, no domain randomization",
        "schedule": "2 s stand + 3 s linear ramp + 30 s hold; metrics after t=10 s",
        "pass": "survive full trial, steady mean body vx >= 80% target, planar RMSE <= max(0.2, 20% target)",
        "scope": "Highest tested speed meeting these conditions, not a global or physical robot speed limit",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    env = Reference(args.source)
    rows = []

    def test(speeds, seeds, hold, stage):
        for speed in speeds:
            for seed in seeds:
                row, _ = env.trial(float(speed), seed, hold)
                row["stage"] = stage
                rows.append(row)
            write_csv(output / "episodes.csv", rows)
            current = [r for r in rows if r["stage"] == stage and abs(r["command_mps"] - speed) < 1e-8]
            print(
                json.dumps(
                    {
                        "stage": stage,
                        "speed": float(speed),
                        "passes": sum(r["passed_tracking"] for r in current),
                        "episodes": len(current),
                        "mean_vx": [r["steady_mean_body_vx_mps"] for r in current],
                    }
                ),
                flush=True,
            )

    coarse = np.arange(0.5, 4.01, 0.5)
    test(coarse, range(70000, 70003), 30, "coarse")

    def all_pass(speed, stage):
        group = [r for r in rows if r["stage"] == stage and abs(r["command_mps"] - speed) < 1e-8]
        return bool(group) and all(r["passed_tracking"] for r in group)

    good = [float(s) for s in coarse if all_pass(s, "coarse")]
    best = max(good) if good else 0.0
    fine = np.round(np.arange(max(0.1, best - 0.2), min(4.0, best + 0.5) + 0.001, 0.1), 2)
    test(fine, range(70100, 70103), 30, "fine")
    good_fine = [float(s) for s in fine if all_pass(s, "fine")]
    candidates = sorted(set(good + good_fine), reverse=True)
    selected = None
    for speed in candidates:
        test([speed], range(71000, 71005), 60, "validation")
        if all_pass(speed, "validation"):
            selected = speed
            break
    summary = {
        "highest_tested_validated_command_mps": selected,
        "validation": [r for r in rows if r["stage"] == "validation"],
        "coarse_best_command_mps": best,
        "note": "No training. Speed limit is conditional on model, startup ramp, duration and finite tested seeds.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    if selected is not None:
        row, trace = env.trial(selected, 71000, 60, str(output / "best-validated.mp4"))
        write_csv(output / "best-trajectory.csv", trace)
        (output / "video-episode.json").write_text(json.dumps(row, indent=2))
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="vendor/unitree_rl_gym")
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
