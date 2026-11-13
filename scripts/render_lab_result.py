"""Render a validated controller on EGL and record the actual OpenGL device."""

import argparse
import json
import time
from pathlib import Path

import mujoco
from OpenGL import GL

from g1lab.lab_evaluation import trial, write_csv
from g1lab.lab_policy import LabReference


def main(args):
    selected = json.loads(Path(args.campaign).read_text())["validated"]
    if selected is None:
        raise ValueError("Campaign has no validated controller to render")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    env = LabReference()
    with mujoco.Renderer(env.model, height=480, width=640):
        gl = {
            name: GL.glGetString(key).decode()
            for name, key in [
                ("vendor", GL.GL_VENDOR),
                ("renderer", GL.GL_RENDERER),
                ("version", GL.GL_VERSION),
            ]
        }
    if args.require_nvidia and "NVIDIA" not in gl["vendor"].upper():
        raise RuntimeError(f"NVIDIA rendering requested but actual GL device is {gl}")
    started = time.perf_counter()
    row, trace = trial(
        env,
        selected["speed"],
        97000,
        hold=30,
        video=str(output / "g1-1660.mp4"),
        heading_hold=True,
        heading_gain=selected["gain"],
        ramp=selected["ramp"],
        diagnostics=True,
    )
    write_csv(output / "trajectory.csv", trace)
    report = {
        "opengl": gl,
        "controller": selected,
        "episode": row,
        "wall_s": time.perf_counter() - started,
        "physics": "CPU MuJoCo",
        "inference": "ONNX Runtime CPU",
        "training_updates": 0,
        "video": "g1-1660.mp4",
        "seed": 97000,
    }
    (output / "render.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, help="Completed controller-search summary.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-nvidia", action="store_true")
    main(parser.parse_args())
