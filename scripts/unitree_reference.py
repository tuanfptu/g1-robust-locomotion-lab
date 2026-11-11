"""Optional vendor reference, NOT a policy trained by this project.

Uses Unitree's 12-leg-DoF model and shipped recurrent policy with their observation/PD contract.
Run from repository root after fetching vendor/unitree_rl_gym. Assets stay outside our source repo.
"""

import argparse
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

from g1lab.terrain import add_terrain

DEFAULT = np.array([-0.1, 0, 0, 0.3, -0.2, 0] * 2)
KP = np.array([100, 100, 100, 150, 40, 40] * 2)
KD = np.array([2, 2, 2, 4, 2, 2] * 2)


def run(args):
    source = Path(args.source).resolve()
    root = ET.parse(source / "resources/robots/g1_description/g1_12dof.xml").getroot()
    root.find("compiler").set("meshdir", str(source / "resources/robots/g1_description/meshes"))
    ET.SubElement(root.find("worldbody"), "light", pos="0 0 5", dir="0 0 -1", directional="true")
    hfield = add_terrain(root, args.terrain, args.difficulty)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)
    model.geom_friction[:, 0] = args.friction
    if hfield:
        rng = np.random.default_rng(args.seed)
        coarse = rng.uniform(0, 1, (17, 17))
        axis = np.linspace(0, 16, 129)
        rows = np.array([np.interp(axis, np.arange(17), row) for row in coarse])
        field = np.array([np.interp(axis, np.arange(17), col) for col in rows.T]).T
        field[61:68, 61:68] = 0
        model.hfield_data[:] = field.ravel()
    policy = torch.jit.load(str(source / "deploy/pre_train/g1/motion.pt"), map_location="cpu")
    policy.eval()
    policy.reset_memory()
    data.qpos[7:] = DEFAULT + np.random.default_rng(args.seed).uniform(-0.005, 0.005, 12)
    mujoco.mj_forward(model, data)
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    action = np.zeros(12)
    command = np.array([args.vx, args.vy, args.yaw])
    target = DEFAULT.copy()
    renderer = mujoco.Renderer(model, height=480, width=640) if args.video else None
    writer = imageio.get_writer(args.video, fps=50) if args.video else None
    viewer = None
    if args.gui:
        from mujoco import viewer as viewer_api

        viewer = viewer_api.launch_passive(model, data)
        viewer.cam.distance = 3
    samples = []
    try:
        for _ in range(round(args.seconds / 0.02)):
            start = time.perf_counter()
            if viewer is not None and not viewer.is_running():
                break
            if args.command_changes:
                phase_index = min(int(data.time / (args.seconds / 4)), 3)
                command = np.array([(0, 0, 0), (0.3, 0, 0), (0.2, 0, 0.3), (0, 0, 0)][phase_index])
            rotation = data.xmat[pelvis].reshape(3, 3)
            phase = data.time % 0.8 / 0.8
            obs = np.concatenate(
                [
                    data.qvel[3:6] * 0.25,
                    rotation.T @ [0, 0, -1],
                    command * [2, 2, 0.25],
                    data.qpos[7:] - DEFAULT,
                    data.qvel[6:] * 0.05,
                    action,
                    [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)],
                ]
            ).astype(np.float32)
            with torch.inference_mode():
                action = policy(torch.from_numpy(obs[None])).numpy()[0]
            target = DEFAULT + action * 0.25
            for _ in range(10):
                data.ctrl[:] = (target - data.qpos[7:]) * KP - data.qvel[6:] * KD
                data.xfrc_applied[:] = 0
                overlap = max(0, min(data.time + 0.002, 5.1) - max(data.time, 5.0))
                if overlap:
                    data.xfrc_applied[pelvis, 1] = args.impulse / 0.1 * overlap / 0.002
                mujoco.mj_step(model, data)
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, pelvis, velocity, 1)
            row = dict(
                time=float(data.time),
                x=float(data.qpos[0]),
                height=float(data.qpos[2]),
                error_sq=float(np.sum((velocity[3:5] - command[:2]) ** 2)),
                yaw_error_sq=float((velocity[2] - command[2]) ** 2),
            )
            samples.append(row)
            if renderer is not None:
                camera = mujoco.MjvCamera()
                camera.lookat[:] = data.qpos[:3]
                camera.distance, camera.azimuth, camera.elevation = 3, 120, -15
                renderer.update_scene(data, camera)
                writer.append_data(renderer.render())
            if viewer is not None:
                with viewer.lock():
                    viewer.cam.lookat[:] = data.qpos[:3]
                viewer.sync()
                time.sleep(max(0, 0.02 - (time.perf_counter() - start)))
            ground = data.qpos[0] * np.tan(np.deg2rad(args.difficulty * 10)) if args.terrain == "slope" else 0
            if data.qpos[2] - ground < 0.45 or data.xmat[pelvis].reshape(3, 3)[2, 2] < 0.45:
                break
    finally:
        for item in (writer, renderer, viewer):
            if item is not None:
                item.close()
    report = {
        "policy_origin": "Unitree shipped pretrained recurrent policy; NOT trained here",
        "robot": "Unitree G1 12 controlled leg joints, fixed upper body",
        "source_commit": "276801e46c5d433564f24658bac64f254b7d2d4b",
        "arguments": vars(args),
        "duration_s": len(samples) * 0.02,
        "survived": len(samples) == round(args.seconds / 0.02),
        "velocity_rmse_mps": float(np.sqrt(np.mean([s["error_sq"] for s in samples]))),
        "yaw_rmse_radps": float(np.sqrt(np.mean([s["yaw_error_sq"] for s in samples]))),
        "x_progress_m": samples[-1]["x"],
    }
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="vendor/unitree_rl_gym")
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--vx", type=float, default=0.3)
    parser.add_argument("--vy", type=float, default=0)
    parser.add_argument("--yaw", type=float, default=0)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--terrain", choices=("flat", "slope", "rough"), default="flat")
    parser.add_argument("--difficulty", type=float, default=0.2)
    parser.add_argument("--friction", type=float, default=0.8)
    parser.add_argument("--impulse", type=float, default=0)
    parser.add_argument("--command-changes", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--video")
    parser.add_argument("--output", required=True)
    torch.set_num_threads(2)
    run(parser.parse_args())
