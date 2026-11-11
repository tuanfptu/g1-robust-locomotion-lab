"""Plot recorded physical trajectories, not reward-only training curves."""

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(args):
    with open(args.input) as handle:
        rows = list(csv.DictReader(handle))
    time = [float(r["time"]) for r in rows]
    legacy_upright = "reward_upright" not in rows[0] and "alive" in rows[0]
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True, layout="constrained")
    axes[0].plot(time, [float(r["command_vx"]) for r in rows], "--", label="Command vx")
    axes[0].plot(time, [float(r["vx"]) for r in rows], label="Measured body vx", linewidth=0.8)
    axes[0].set_ylabel("Velocity (m/s)")
    for side in ("left", "right"):
        axes[1].plot(
            time,
            [100 * float(r[f"{side}_foot_clearance_m"]) for r in rows],
            label=f"{side} foot",
            linewidth=0.8,
        )
    axes[1].set_ylabel("Foot site clearance (cm)")
    axes[2].plot(
        time,
        [float(r["upright"]) * (2 if legacy_upright else 1) for r in rows],
        label="Pelvis vertical-axis Z",
    )
    if legacy_upright:
        print("Legacy log: reconstructed physical upright = 2 * reward_upright; raw CSV unchanged.")
    axes[2].set_ylabel("Upright component")
    axes[2].set_ylim(0.4, 1.05)
    axes[2].set_xlabel("Simulation time (s)")
    for axis in axes:
        axis.legend(loc="lower right")
        axis.grid(alpha=0.2)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("output")
    main(parser.parse_args())
