"""Speed-and-stop audit of the original Unitree RL Lab full-body ONNX policy."""

import argparse
import hashlib
import json
from pathlib import Path

from g1lab.fast_plan import LAB_REVISION, MUJOCO_REVISION
from g1lab.lab_evaluation import trial, write_csv
from g1lab.lab_policy import LabReference


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
