"""Render comparable evaluation reports without inventing missing results."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def report(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    summaries = [json.loads((Path(p) / "summary.json").read_text()) for p in args.inputs]
    protocols = [json.loads((Path(p) / "protocol.json").read_text()) for p in args.inputs]
    for key in ("suite", "episodes", "seconds", "seed"):
        if len({p[key] for p in protocols}) != 1:
            raise ValueError(f"Incompatible evaluation protocols: {key}")
    scenarios = sorted(set.intersection(*(set(s) for s in summaries)))
    if not scenarios:
        raise ValueError("No shared scenarios to compare")
    labels = [Path(p).name for p in args.inputs]
    fig, axes = plt.subplots(2, 1, figsize=(max(9, len(scenarios) * 0.65), 9), layout="constrained")
    x = np.arange(len(scenarios))
    width = 0.8 / len(summaries)
    for index, (label, summary) in enumerate(zip(labels, summaries)):
        positions = x - 0.4 + width / 2 + index * width
        axes[0].bar(
            positions,
            [summary[s]["velocity_rmse_mps"]["mean"] for s in scenarios],
            width,
            label=label,
            yerr=[summary[s]["velocity_rmse_mps"]["std"] for s in scenarios],
        )
        axes[1].bar(positions, [summary[s]["survival_rate"] for s in scenarios], width, label=label)
    axes[0].set_ylabel("Velocity RMSE (m/s), mean ± episode SD")
    axes[1].set_ylabel("Survival fraction")
    axes[1].set_ylim(0, 1.05)
    for ax in axes:
        ax.set_xticks(x, scenarios, rotation=65, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    text = [
        "# Evaluation comparison",
        "",
        "Episode SD is not uncertainty across training seeds.",
        "Compare only runs with matching protocol.json settings and different output/bundle paths.",
        "",
        "| Run | Scenario | Velocity RMSE (m/s) | Survival |",
        "|---|---|---:|---:|",
    ]
    for label, summary in zip(labels, summaries):
        for scenario in scenarios:
            row = summary[scenario]
            text.append(
                f"| {label} | {scenario} | {row['velocity_rmse_mps']['mean']:.3f} | {row['survival_rate']:.1%} |"
            )
    (output / "report.md").write_text("\n".join(text) + "\n")
    print(output / "comparison.png")
