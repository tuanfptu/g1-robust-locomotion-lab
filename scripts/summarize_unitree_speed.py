"""Summarize the finite-condition original-policy speed benchmark."""

import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    source = Path("artifacts/unitree-original-speed-v1")
    destination = Path("docs/experiment-data/unitree-original-speed-v1")
    destination.mkdir(parents=True, exist_ok=True)
    for name in ["protocol.json", "summary.json", "episodes.csv", "video-episode.json"]:
        shutil.copy2(source / name, destination / name)
    with (source / "episodes.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    summary = json.loads((source / "summary.json").read_text())
    validated = summary["highest_tested_validated_command_mps"]
    validation = [r for r in rows if r["stage"] == "validation" and float(r["command_mps"]) == validated]
    actual = np.mean([float(r["steady_mean_body_vx_mps"]) for r in validation])
    rmse = np.mean([float(r["steady_rmse_mps"]) for r in validation])
    groups = {}
    for row in rows:
        if row["stage"] == "validation":
            continue
        speed = float(row["command_mps"])
        if speed not in groups or row["stage"] == "fine" and groups[speed][0]["stage"] == "coarse":
            groups[speed] = []
        if not groups[speed] or row["stage"] == groups[speed][0]["stage"]:
            groups[speed].append(row)
    speeds = sorted(groups)
    measured = [
        np.mean([float(r["steady_mean_body_vx_mps"]) for r in groups[s] if r["steady_mean_body_vx_mps"]])
        if any(r["steady_mean_body_vx_mps"] for r in groups[s])
        else np.nan
        for s in speeds
    ]
    survival = [sum(r["survived"] == "True" for r in groups[s]) / len(groups[s]) for s in speeds]
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(speeds, speeds, "--", color="gray", label="Ideal tracking")
    axes[0].plot(speeds, measured, "o-", label="Measured body-forward mean")
    axes[0].set_ylabel("Forward speed (m/s)")
    axes[0].legend()
    axes[0].set_title("Original Unitree G1 policy; 3 trials per command, flat ground")
    axes[1].plot(speeds, survival, "o-", color="darkorange")
    axes[1].set_ylabel("Survival fraction")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_xlabel("Commanded forward speed (m/s)")
    fig.tight_layout()
    fig.savefig("docs/images/unitree-speed.png", dpi=160)
    plt.close(fig)
    lines = [
        "# Original Unitree G1 policy: measured speed envelope",
        "",
        f"Highest tested command meeting the defined tracking gate and 5/5 fresh-seed 60-second holds: **{validated:.1f} m/s**. "
        f"Measured steady body-forward speed: **{actual:.3f} m/s** ({actual * 3.6:.2f} km/h). "
        f"Mean planar tracking RMSE: {rmse:.3f} m/s. This is a conditional tested operating point, not a global maximum.",
        "",
        "![Speed sweep](images/unitree-speed.png)",
        "",
        "## Protocol",
        "",
        "No training or fine-tuning. Original Unitree `motion.pt`, pinned source commit "
        "`276801e46c5d433564f24658bac64f254b7d2d4b`, original Unitree 12-controlled-leg-DoF model with fixed upper body. "
        "This differs from the project’s Menagerie 29-actuator student model. Physics step 0.002 s, control step 0.02 s, "
        "official observation/action scaling, phase and PD gains. Terrain is flat, sliding friction 0.8; no pushes or domain randomization.",
        "",
        "Each trial starts standing for 2 s, ramps the command over 3 s, then holds it for 30 s. "
        "Coarse commands: 0.5–4.0 m/s in 0.5 increments, seeds 70000–70002. "
        "Refinement: 1.3–2.0 m/s in 0.1 increments, seeds 70100–70102. "
        "Steady metrics exclude the first 10 s. Pass requires full survival, steady mean body-forward speed at least "
        "80% of command, and planar RMSE no greater than max(0.2 m/s, 20% of command). "
        "Fall: pelvis height below 0.45 m, upright cosine below 0.45, or non-finite state. "
        "The selected command is checked with 60-second holds at fresh seeds 71000–71004. "
        "Seeds only change initial joint positions within ±0.005 rad; they are not diverse disturbances.",
        "",
        "## Results",
        "",
        "| Command | Survived / 3 | Tracking pass / 3 | Mean body vx | Mean world vx |",
        "|---:|---:|---:|---:|---:|",
    ]
    for speed in speeds:
        group = groups[speed]
        body = [float(r["steady_mean_body_vx_mps"]) for r in group if r["steady_mean_body_vx_mps"]]
        world = [float(r["steady_mean_world_vx_mps"]) for r in group if r["steady_mean_world_vx_mps"]]
        a = f"{np.mean(body):.3f}" if body else "—"
        b = f"{np.mean(world):.3f}" if world else "—"
        lines.append(
            f"| {speed:.1f} | {sum(r['survived'] == 'True' for r in group)}/3 | {sum(r['passed_tracking'] == 'True' for r in group)}/3 | {a} | {b} |"
        )
    lines += [
        "",
        "## Limits and interpretation",
        "",
        "At command 2.0 m/s, the robot survives the 30-second hold but averages only about 1.50 m/s "
        "in its own forward axis and drifts/turns strongly; world-x velocity can become negative. "
        "It is not successful straight-line tracking. Commands of 2.5 m/s and above fell in all tested coarse trials. "
        "Transient peaks before falling are not treated as sustainable speed. "
        "A 0.1 m/s refinement cannot identify an exact boundary between adjacent commands. "
        "Survival and forward tracking do not by themselves certify a running gait with a flight phase. "
        "No conclusion is made about physical G1 hardware limits, other Unitree policies, terrain robustness, "
        "abrupt acceleration, or performance on the project’s different robot model.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "MUJOCO_GL=egl EGL_PLATFORM=surfaceless MPLCONFIGDIR=/tmp/g1lab-mpl \\",
        "  .venv/bin/python scripts/benchmark_unitree_speed.py \\",
        "  --output artifacts/unitree-original-speed-v1",
        "```",
        "",
        "Fetch the optional pinned Unitree assets first using `scripts/fetch_teacher.sh`. "
        "Use a new output directory on rerun. The best-validated real-time MP4 and raw trajectory remain local; "
        "compact provenance and all per-trial results are retained under `experiment-data/unitree-original-speed-v1/`.",
        "",
    ]
    Path("docs/UNITREE_SPEED.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
