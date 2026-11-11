"""Publish compact, paired local-pilot results without model weights or trajectories."""

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path("runs/mixed-1660-v1")
    destination = Path("docs/experiment-data/mixed-1660-v1")
    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "progress.json",
        "protocol.json",
        "selection.json",
        "heldout.json",
        "baseline-heldout.json",
        "input-sha256.json",
    ):
        shutil.copy2(root / name, destination / name)
    for name in ("baseline-full", "selected-full"):
        folder = destination / name
        folder.mkdir(exist_ok=True)
        for file in ("summary.json", "episodes.csv", "protocol.json", "environments.json"):
            shutil.copy2(root / name / file, folder / file)
    timings = [json.loads((root / f"round-{i}/training.json").read_text()) for i in range(3)]
    (destination / "training.json").write_text(json.dumps(timings, indent=2))
    baseline = json.loads((root / "baseline-heldout.json").read_text())
    candidate = json.loads((root / "heldout.json").read_text())
    fig, ax = plt.subplots(figsize=(7, 4))
    positions = np.arange(3)
    for delta, data, label in ((-0.18, baseline, "Previous policy"), (0.18, candidate, "Mixed PPO + BC")):
        ax.bar(
            positions + delta,
            [data[t]["velocity_rmse_mps"]["mean"] for t in data],
            0.36,
            yerr=[data[t]["velocity_rmse_mps"]["std"] for t in data],
            capsize=4,
            label=label,
        )
    ax.set_xticks(positions, ["Flat", "Slope 2°", "Rough ~1.2 cm"])
    ax.set_ylabel("Velocity RMSE (m/s); lower is better")
    ax.set_title("Paired seeds 40000–40002, 30 s, command 0.3 m/s")
    ax.legend()
    fig.tight_layout()
    fig.savefig("docs/images/mixed-1660.png", dpi=160)
    plt.close(fig)
    lines = [
        "# Mixed-terrain local pilot — 1660 Ti machine",
        "",
        "One training trajectory, initialized from the previous Unitree-teacher/DAgger/PPO+BC policy. "
        "This is an interleaved PPO + BC pilot, not from-scratch locomotion or a multi-seed algorithm comparison.",
        "",
        f"Trained {sum(t['additional_timesteps'] for t in timings):,} additional PPO transitions. "
        f"Measured throughput: {min(t['steps_per_second'] for t in timings):.0f}–"
        f"{max(t['steps_per_second'] for t in timings):.0f} transitions/s. CPU physics and network, GPU rendering.",
        "",
        "![Paired velocity tracking](images/mixed-1660.png)",
        "",
        "## Held-out terrain test",
        "",
        "Same seeds 40000–40002, 30 s each, command 0.3 m/s. Mean RMSE; survival shown as completed episodes. "
        "Error bars show episode standard deviation, not confidence over independent training runs.",
        "",
        "| Terrain | Previous survival | Mixed survival | Previous RMSE | Mixed RMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for terrain in baseline:
        old, new = baseline[terrain], candidate[terrain]
        lines.append(
            f"| {terrain} | {round(old['survival_rate'] * 3)}/3 | {round(new['survival_rate'] * 3)}/3 | "
            f"{old['velocity_rmse_mps']['mean']:.4f} | {new['velocity_rmse_mps']['mean']:.4f} |"
        )
    lines += [
        "",
        "## Broader regression audit",
        "",
        "The following suite uses identical seeds 41000–41002 and 30 s episodes. Forward command is "
        "0.4 m/s here, unlike the 0.3 m/s terrain test. These data were measured after checkpoint selection.",
        "",
        "| Scenario | Previous survival | Mixed survival | Previous velocity RMSE | Mixed velocity RMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    old = json.loads((root / "baseline-full/summary.json").read_text())
    new = json.loads((root / "selected-full/summary.json").read_text())
    for name in old:
        a, b = old[name], new[name]
        lines.append(
            f"| {name} | {round(a['survival_rate'] * 3)}/3 | {round(b['survival_rate'] * 3)}/3 | "
            f"{a['velocity_rmse_mps']['mean']:.3f} | {b['velocity_rmse_mps']['mean']:.3f} |"
        )
    lines += [
        "",
        "## Interpretation and reproduction",
        "",
        "Selection used seeds 6000–6001 only; two candidate rounds were accepted and the third rejected. "
        "The terrain-selected candidate is `runs/mixed-1660-v1/selected`. The broader audit found loss of "
        "in-place turning (0/3 completions versus 3/3 for the previous policy), so it is NOT promoted to "
        "the default demonstration. This is a negative result for the narrow selection gate. "
        "The previous bundle is retained. "
        "No model is certified for arbitrary terrain; three test episodes cannot establish reliability. "
        "Improvements on a single task do not imply improvements on other commands or terrain levels.",
        "",
        "See [local training protocol](LOCAL_TRAINING.md) for the command, selection heuristic, "
        "teacher provenance, data dependencies, and criteria for moving to a larger machine.",
        "",
    ]
    Path("docs/EXPERIMENT_MIXED_1660.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
