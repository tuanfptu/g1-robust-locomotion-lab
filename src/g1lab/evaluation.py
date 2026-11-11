"""Deterministic held-out scenarios with episode and trajectory exports."""

import csv
import json
from dataclasses import replace
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from .config import EnvConfig
from .env import G1Env
from .training import load_policy, policy_action


def scenarios(suite, seconds):
    common = EnvConfig(random_commands=False, episode_seconds=seconds)
    cases = [
        ("stand", replace(common, command=(0, 0, 0))),
        ("forward", common),
        ("lateral", replace(common, command=(0, 0.2, 0))),
        ("turn", replace(common, command=(0, 0, 0.5))),
        ("command_changes", common),
    ]
    if suite == "full":
        cases += [(f"friction_{mu}", replace(common, friction=mu)) for mu in (0.3, 0.6, 1.1)]
        cases += [(f"mass_{mass}", replace(common, mass_scale=mass)) for mass in (0.85, 1.15)]
        cases += [
            (f"{terrain}_{level}", replace(common, terrain=terrain, difficulty=level))
            for terrain in ("slope", "rough", "stairs")
            for level in (0.2, 0.5)
        ]
        cases += [
            (
                f"push_{impulse}_{direction}",
                replace(common, push_impulse=impulse, push_angle=angle, push_at=min(5.0, seconds / 3)),
            )
            for impulse in (2, 5, 10)
            for direction, angle in (("front", 0), ("side", np.pi / 2))
        ]
    return cases


def write_csv(path, rows):
    if rows:
        with Path(path).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def evaluate(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_text(json.dumps(vars(args), indent=2))
    records = []
    for name, config in scenarios(args.suite, args.seconds):
        if args.bundle:
            model, vec, raw = load_policy(args.bundle, config)
        else:
            raw, model, vec = G1Env(config), None, None
        try:
            for episode in range(args.episodes):
                seed = args.seed + episode
                obs, _ = raw.reset(seed=seed)
                trajectory = []
                writer = (
                    imageio.get_writer(output / f"{name}-{seed}.mp4", fps=raw.metadata["render_fps"])
                    if args.video and episode == 0
                    else None
                )
                total = 0.0
                recovery_start, recovery_time = None, None
                impulse_total = 0.0
                fell = False
                try:
                    while True:
                        if name == "command_changes":
                            phase = min(int(raw.data.time / max(config.episode_seconds / 4, 0.02)), 3)
                            raw.set_command(*[(0, 0, 0), (0.5, 0, 0), (0.2, 0, 0.5), (0, 0, 0)][phase])
                            obs = raw._observation()
                        action = (
                            policy_action(model, vec, obs) if model is not None else np.zeros(raw.model.nu)
                        )
                        obs, reward, terminated, truncated, info = raw.step(action)
                        trajectory.append({"scenario": name, "seed": seed, **info})
                        total += reward
                        impulse_total += info["impulse_ns"]
                        if config.push_impulse > 0 and info["time"] >= config.push_at + config.push_duration:
                            recovered = info["velocity_error_sq"] < 0.15**2 and info["upright"] > 0.9
                            if recovered:
                                if recovery_start is None:
                                    recovery_start = info["time"]
                                if info["time"] - recovery_start >= 0.5 and recovery_time is None:
                                    recovery_time = recovery_start - (config.push_at + config.push_duration)
                            else:
                                recovery_start = None
                        if writer:
                            writer.append_data(raw.render())
                        if terminated or truncated:
                            fell = terminated
                            break
                finally:
                    if writer:
                        writer.close()
                duration = len(trajectory) * config.control_dt
                energy = sum(row["energy_j"] for row in trajectory)
                distance = sum(row["distance_m"] for row in trajectory)
                contact_samples = sum(row["contacting_feet"] for row in trajectory)
                records.append(
                    {
                        "scenario": name,
                        "seed": seed,
                        "return": total,
                        "duration_s": duration,
                        "survived": not fell,
                        "velocity_rmse_mps": float(
                            np.sqrt(np.mean([r["velocity_error_sq"] for r in trajectory]))
                        ),
                        "yaw_rmse_radps": float(np.sqrt(np.mean([r["yaw_error_sq"] for r in trajectory]))),
                        "energy_j": energy,
                        "distance_m": distance,
                        "energy_per_meter": energy / distance if distance > 0.1 else None,
                        "mean_power_w": energy / duration,
                        "x_progress_m": info["x_progress_m"],
                        "contact_foot_slip_mps": sum(r["foot_slip_sum_mps"] for r in trajectory)
                        / contact_samples
                        if contact_samples
                        else None,
                        "delivered_impulse_ns": impulse_total,
                        "push_received": impulse_total > 0,
                        "recovery_time_s": recovery_time,
                    }
                )
                write_csv(output / f"{name}-{seed}.csv", trajectory)
        finally:
            vec.close() if vec is not None else raw.close()
    write_csv(output / "episodes.csv", records)
    summary = {}
    for name, _ in scenarios(args.suite, args.seconds):
        rows = [r for r in records if r["scenario"] == name]
        metrics = {}
        for key in ("velocity_rmse_mps", "yaw_rmse_radps", "duration_s", "mean_power_w"):
            values = [r[key] for r in rows]
            metrics[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
        metrics["survival_rate"] = float(np.mean([r["survived"] for r in rows]))
        metrics["push_received_count"] = sum(r["push_received"] for r in rows)
        metrics["episodes"] = len(rows)
        summary[name] = metrics
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
