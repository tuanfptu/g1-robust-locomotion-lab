"""Summarize the six-task preservation follow-up and retain compact evidence."""

import hashlib
import json
import shutil
from pathlib import Path


def main():
    root = Path("runs/preservation-1660-v2")
    output = Path("docs/experiment-data/preservation-1660-v2")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("protocol.json", "progress.json", "selection.json", "heldout.json", "baseline-heldout.json"):
        shutil.copy2(root / name, output / name)
    for group in ("heldout", "baseline-heldout"):
        for folder in (root / group).iterdir():
            if folder.is_dir():
                destination = output / group / folder.name
                destination.mkdir(parents=True, exist_ok=True)
                for name in ("episodes.csv", "protocol.json", "environments.json"):
                    shutil.copy2(folder / name, destination / name)
    hashes = {
        name: hashlib.sha256((root / "selected" / name).read_bytes()).hexdigest()
        for name in ("policy.zip", "vecnormalize.pkl")
    }
    (output / "selected-sha256.json").write_text(json.dumps(hashes, indent=2))
    timing = []
    for folder in sorted(root.glob("round-*")):
        if folder.is_dir():
            timing.append(json.loads((folder / "training.json").read_text()))
    (output / "training.json").write_text(json.dumps(timing, indent=2))
    old = json.loads((root / "baseline-heldout.json").read_text())
    new = json.loads((root / "heldout.json").read_text())
    selected = json.loads((root / "selection.json").read_text())["source"]
    lines = [
        "# Six-task skill-preservation pilot",
        "",
        "Follow-up to the turning regression in [the terrain-only pilot](EXPERIMENT_MIXED_1660.md). "
        "One shared policy is trained with PPO and behavioral replay from the previous student. "
        "The student itself originated from a Unitree teacher; this is not from-scratch learning.",
        "",
        f"Selected source: `{selected}`. Additional PPO transitions: "
        f"{sum(t['additional_timesteps'] for t in timing):,}. "
        f"Measured CPU training throughput: {min(t['steps_per_second'] for t in timing):.0f}–"
        f"{max(t['steps_per_second'] for t in timing):.0f} transitions/s.",
        "",
        "## Paired held-out results",
        "",
        "Seeds 50000–50002; 30 seconds per episode. All forward terrain commands are 0.3 m/s. "
        "Lateral command: 0.2 m/s; turn: 0.5 rad/s. Three episodes and one training trajectory "
        "are insufficient to claim statistical superiority or deployment reliability.",
        "",
        "| Task | Previous survival | Selected survival | Previous velocity RMSE | Selected velocity RMSE | Previous yaw RMSE | Selected yaw RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task in old:
        a, b = old[task], new[task]
        lines.append(
            f"| {task} | {round(a['survival_rate'] * 3)}/3 | {round(b['survival_rate'] * 3)}/3 | "
            f"{a['velocity_rmse_mps']['mean']:.4f} | {b['velocity_rmse_mps']['mean']:.4f} | "
            f"{a['yaw_rmse_radps']['mean']:.4f} | {b['yaw_rmse_radps']['mean']:.4f} |"
        )
    lines += [
        "",
        "## Scope",
        "",
        "The candidate completed only 2/3 turning episodes versus 3/3 for the prior policy. "
        "It is NOT promoted. The short 15-second validation gate missed a later instability. "
        "The script now defaults to two 30-second validation episodes for future campaigns; this "
        "stronger setting has not been retrospectively presented as part of the completed pilot.",
        "",
        "The broader 22-scenario paired audit in the previous report applies to the previous policy and "
        "the first terrain-only candidate; it does not certify this follow-up. Steeper slopes, taller "
        "roughness, stairs, arbitrary command sequences and new push conditions remain outside this "
        "six-task test. The previous default checkpoint is retained.",
        "",
        "Validation seed 7000 was used for model selection. Test seeds 50000–50002 were used only after "
        "selection. No further policy training was performed using these results. See [the local protocol](LOCAL_TRAINING.md) "
        "for replay collection, acceptance rules, and reproduction commands.",
        "",
    ]
    Path("docs/EXPERIMENT_PRESERVATION_1660.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
