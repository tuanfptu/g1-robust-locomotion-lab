"""Render independent CPU MuJoCo worlds together; inference only, no PPO updates."""

import argparse
import csv
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from benchmark_unitree_speed import Reference
from PIL import Image, ImageDraw, ImageFont
from unitree_reference import DEFAULT, KD, KP

from g1lab.cli import positive, positive_float


class World:
    def __init__(self, reference, source, index, count):
        self.model = reference.model  # Read-only shared model, separate mutable MjData and LSTM per world.
        self.data = mujoco.MjData(self.model)
        self.policy = torch.jit.load(str(Path(source) / "deploy/pre_train/g1/motion.pt"), map_location="cpu")
        self.policy.eval()
        self.pelvis = reference.pelvis
        self.index = index
        self.target_speed = [0.5, 1.0, 1.6, 2.5][index % 4]
        side = int(np.ceil(np.sqrt(count)))
        self.offset = np.array(
            [(index % side - (side - 1) / 2) * 3.0, (index // side - (side - 1) / 2) * 3.0]
        )
        self.rng = np.random.default_rng(88000 + index)
        self.resets = self.falls = self.steps = 0
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:2] = self.offset
        self.data.qpos[7:] = DEFAULT + self.rng.uniform(-0.005, 0.005, 12)
        self.policy.reset_memory()
        self.action = np.zeros(12)
        self.wait = 0
        self.fallen = False
        mujoco.mj_forward(self.model, self.data)

    def step(self):
        if self.wait == 1 or self.data.time >= 8.0 + (self.index % 4) * 0.4:
            self.resets += 1
            self.reset()
        if self.wait > 0:
            self.wait -= 1
            self.action[:] = 0
        else:
            rotation = self.data.xmat[self.pelvis].reshape(3, 3)
            phase = 2 * np.pi * (self.data.time % 0.8) / 0.8
            command = self.target_speed * np.clip((self.data.time - 1.0) / 2.0, 0, 1)
            obs = np.concatenate(
                [
                    self.data.qvel[3:6] * 0.25,
                    rotation.T @ [0, 0, -1],
                    [command * 2, 0, 0],
                    self.data.qpos[7:] - DEFAULT,
                    self.data.qvel[6:] * 0.05,
                    self.action,
                    [np.sin(phase), np.cos(phase)],
                ]
            ).astype(np.float32)
            with torch.inference_mode():
                self.action = self.policy(torch.from_numpy(obs[None])).numpy()[0]
        target = DEFAULT + self.action * 0.25
        for _ in range(10):
            self.data.ctrl[:] = (target - self.data.qpos[7:]) * KP - self.data.qvel[6:] * KD
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        finite = np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
        fallen = bool(
            self.data.qpos[2] < 0.45 or self.data.xmat[self.pelvis].reshape(3, 3)[2, 2] < 0.45 or not finite
        )
        if fallen and not self.fallen:
            self.falls += 1
            self.wait = 26  # Show the fall for about half a simulated second before resetting this world.
        self.fallen = fallen
        if not finite:
            self.resets += 1
            self.reset()
        return self.fallen


def main(args):
    torch.set_num_threads(1)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    reference = Reference(args.source)
    ground = mujoco.mj_name2id(reference.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    reference.model.geom_size[ground, :2] = 200
    # Offscreen framebuffer only; no dynamics parameters are changed for the overview.
    reference.model.vis.global_.offwidth = 1280
    reference.model.vis.global_.offheight = 720
    worlds = [World(reference, args.source, i, args.worlds) for i in range(args.worlds)]
    renderer = mujoco.Renderer(reference.model, height=720, width=1280, max_geom=20000)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    option, perturb = mujoco.MjvOption(), mujoco.MjvPerturb()
    colors = np.array([[0.3, 0.8, 1.0], [0.4, 1.0, 0.5], [1.0, 0.8, 0.2], [1.0, 0.35, 0.3]])
    rows = []
    simulation_seconds = 0.0
    started = time.perf_counter()
    frames = 0
    try:
        with (
            ThreadPoolExecutor(max_workers=args.workers) as pool,
            imageio.get_writer(output / "swarm.mp4", fps=25) as writer,
        ):
            for step in range(max(2, 2 * round(args.seconds / 0.04))):
                tick = time.perf_counter()
                list(pool.map(lambda world: world.step(), worlds))
                simulation_seconds += time.perf_counter() - tick
                if (step + 1) % 2:
                    continue
                positions = np.array([w.data.qpos[:3] for w in worlds])
                camera = mujoco.MjvCamera()
                center = (positions[:, :2].min(axis=0) + positions[:, :2].max(axis=0)) / 2
                camera.lookat[:] = [*center, 0.5]
                extent = max(float(np.ptp(positions[:, 0])) + 3, float(np.ptp(positions[:, 1])) + 3)
                camera.distance = max(18.0, extent * 1.45)
                camera.azimuth, camera.elevation = 125, -50
                renderer.update_scene(worlds[0].data, camera, scene_option=option)
                for i, world in enumerate(worlds):
                    if i == 0:
                        start, end = 0, renderer.scene.ngeom
                    else:
                        start = renderer.scene.ngeom
                        mujoco.mjv_addGeoms(
                            reference.model,
                            world.data,
                            option,
                            perturb,
                            mujoco.mjtCatBit.mjCAT_DYNAMIC,
                            renderer.scene,
                        )
                        end = renderer.scene.ngeom
                    for j in range(start, end):
                        geom = renderer.scene.geoms[j]
                        if geom.category & int(mujoco.mjtCatBit.mjCAT_DYNAMIC):
                            geom.rgba[:3] = 0.35 * geom.rgba[:3] + 0.65 * colors[i % 4]
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 1280, 109), fill=(10, 15, 24))
                resets, falls = sum(w.resets for w in worlds), sum(w.falls for w in worlds)
                draw.text(
                    (15, 8),
                    f"{args.worlds} INDEPENDENT G1 WORLDS | CPU physics / threaded stepping",
                    font=font,
                    fill="white",
                )
                draw.text(
                    (15, 39),
                    f"Simulated t={(step + 1) * 0.02:.1f}s | falls={falls} | individual resets={resets}",
                    font=small,
                    fill="#9ee8ff",
                )
                draw.text(
                    (15, 63),
                    "Blue 0.5 | green 1.0 | yellow 1.6 | red 2.5 m/s target (stress test)",
                    font=small,
                    fill="white",
                )
                draw.text(
                    (15, 85),
                    "Original Unitree policy inference only. No learning. Independent worlds do not collide.",
                    font=small,
                    fill="#ffcf7b",
                )
                writer.append_data(np.array(frame))
                frames += 1
                rows.append(
                    {
                        "time_s": (step + 1) * 0.02,
                        "falls": falls,
                        "resets": resets,
                        "standing_worlds": sum(not w.fallen for w in worlds),
                    }
                )
                if frames % 50 == 0:
                    print(json.dumps(rows[-1]), flush=True)
    finally:
        renderer.close()
    wall = time.perf_counter() - started
    transitions = sum(w.steps for w in worlds)
    summary = {
        "worlds": args.worlds,
        "workers": args.workers,
        "simulated_seconds_per_world": transitions * 0.02 / args.worlds,
        "wall_seconds_including_render": wall,
        "simulation_and_policy_wall_seconds": simulation_seconds,
        "aggregate_control_steps_per_second_without_render": transitions / simulation_seconds,
        "aggregate_control_steps_per_second_including_render": transitions / wall,
        "per_world_realtime_factor_including_render": transitions * 0.02 / args.worlds / wall,
        "falls": sum(w.falls for w in worlds),
        "resets": sum(w.resets for w in worlds),
        "frames": frames,
        "video_fps": 25,
        "learning_updates": 0,
        "backend": "Native MuJoCo CPU, separate MjData and separate LSTM state per world, shared read-only model",
        "note": "Video plays simulation time at 1x; rendering/collection can take longer than video duration.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "protocol.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "seed_base": 88000,
                "policy_sha256": hashlib.sha256(
                    (Path(args.source) / "deploy/pre_train/g1/motion.pt").read_bytes()
                ).hexdigest(),
                "source_commit": "276801e46c5d433564f24658bac64f254b7d2d4b",
                "commands": [0.5, 1.0, 1.6, 2.5],
                "physics_dt": 0.002,
                "control_dt": 0.02,
                "reset": "independently on fall or 8–9.2 simulated seconds",
                "rendering": "All independent states are combined in one overview; no shared-world robot interactions",
            },
            indent=2,
        )
    )
    with (output / "timeline.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="vendor/unitree_rl_gym")
    parser.add_argument("--output", required=True)
    parser.add_argument("--worlds", type=positive, default=16)
    parser.add_argument("--workers", type=positive, default=4)
    parser.add_argument("--seconds", type=positive_float, default=12)
    main(parser.parse_args())
