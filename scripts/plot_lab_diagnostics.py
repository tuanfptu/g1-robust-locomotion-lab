"""Plot measured diagnostics from the finite controller campaign."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(args):
    source = Path(args.campaign)
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
    for speed in [1.5, 1.6, 1.65]:
        path = source / f"development-v{speed:g}-g2-r3-s94000.csv"
        with path.open() as handle:
            rows = list(csv.DictReader(handle))

        def values(key):
            return np.array([float(row[key]) for row in rows])

        t = values("time_s")
        label = f"command {speed:g} m/s, gain 2, ramp 3 s"
        axes[0].plot(t, values("vx_mps"), label=label, linewidth=0.8)
        axes[1].plot(t, values("yaw_rate"), linewidth=0.8)
        axes[2].plot(t, values("peak_torque_utilization"), linewidth=0.8)
        axes[3].plot(t, values("contact_foot_speed_mps"), linewidth=0.8)
    axes[0].set_ylabel("Body vx (m/s)")
    axes[0].legend(loc="upper right")
    axes[1].set_ylabel("Body yaw rate (rad/s)")
    axes[2].set_ylabel("Peak torque / XML limit")
    axes[2].axhline(1, color="black", linestyle="--", linewidth=0.8)
    axes[3].set_ylabel("Contact foot XY speed\nproxy (m/s)")
    axes[3].set_xlabel("Simulation time (s)")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle("G1 public ONNX: measured oscillation near speed limit; no RL updates")
    fig.savefig(args.output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
