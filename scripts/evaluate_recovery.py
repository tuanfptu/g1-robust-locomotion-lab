"""Recovery-specific success metrics; no locomotion survival-rate substitution."""

import argparse
import csv
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from g1lab.cli import positive
from g1lab.config import EnvConfig
from g1lab.env import G1Env
from g1lab.training import load_policy, policy_action


def main(args):
    torch.set_num_threads(1)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config = EnvConfig(**json.loads(Path(args.config).read_text()))
    if config.task != "recovery":
        raise ValueError("Use a recovery task config")
    if args.bundle:
        model, vec, raw = load_policy(args.bundle, config)
    else:
        model, vec, raw = None, None, G1Env(config)
    if raw.config.task != "recovery":
        raise ValueError("Checkpoint is not a recovery policy")
    (output / "protocol.json").write_text(json.dumps(vars(args), indent=2))
    (output / "environment.json").write_text(json.dumps(raw.config.to_dict(), indent=2))
    rows = []
    writer = None
    try:
        for episode in range(args.episodes):
            obs, _ = raw.reset(seed=args.seed + episode)
            if args.video and episode == 0:
                raw.render_mode = "rgb_array"
                writer = imageio.get_writer(output / "recovery.mp4", fps=raw.metadata["render_fps"])
            max_height, energy, total_reward, max_joint_speed = 0.0, 0.0, 0.0, 0.0
            while True:
                action = policy_action(model, vec, obs) if model else np.zeros(raw.model.nu)
                obs, reward, done, truncated, info = raw.step(action)
                if writer is not None:
                    writer.append_data(raw.render())
                max_height = max(max_height, info["base_height_m"])
                energy += info["energy_j"]
                total_reward += reward
                max_joint_speed = max(max_joint_speed, float(np.max(np.abs(raw.data.qvel[raw.vadr]))))
                if done or truncated:
                    break
            if writer is not None:
                writer.close()
                writer = None
            rows.append(
                {
                    "seed": args.seed + episode,
                    "success": info["recovery_success"],
                    "duration_s": info["time"],
                    "recovery_time_s": info["time"] if info["recovery_success"] else None,
                    "max_pelvis_height_m": max_height,
                    "final_upright_cos": info["upright"],
                    "max_joint_speed_rad_s": max_joint_speed,
                    "mechanical_energy_j": energy,
                    "return": total_reward,
                }
            )
    finally:
        if writer is not None:
            writer.close()
        vec.close() if vec else raw.close()
    with (output / "episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    successes = [row["recovery_time_s"] for row in rows if row["success"]]
    summary = {
        "episodes": len(rows),
        "success_rate": len(successes) / len(rows),
        "mean_success_time_s": float(np.mean(successes)) if successes else None,
        "mean_max_pelvis_height_m": float(np.mean([r["max_pelvis_height_m"] for r in rows])),
        "mean_return": float(np.mean([r["return"] for r in rows])),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle")
    parser.add_argument("--config", default="configs/recovery-supine.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=positive, default=10)
    parser.add_argument("--seed", type=int, default=60000)
    parser.add_argument("--video", action="store_true")
    main(parser.parse_args())
