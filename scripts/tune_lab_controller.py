"""Finite controller search with separate held-out validation; no weight training."""

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np

from g1lab.fast_plan import LAB_REVISION, MUJOCO_REVISION
from g1lab.lab_evaluation import trial, write_csv
from g1lab.lab_policy import LabReference


def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    env = LabReference()
    protocol = {
        "lab_revision": LAB_REVISION,
        "mujoco_revision": MUJOCO_REVISION,
        "policy_sha256": hashlib.sha256(env.policy_path.read_bytes()).hexdigest(),
        "weights_updated": False,
        "development": {
            "speeds": [1.5, 1.6, 1.65],
            "gains": [1.0, 2.0, 4.0],
            "ramps": [3.0, 5.0],
            "seeds": [94000, 94001],
            "hold_s": 30,
        },
        "validation": {"seeds": list(range(95000, 95010)), "hold_s": 60},
        "selection": "all development seeds pass; highest command, then lowest mean yaw RMS",
        "fallback": "if selected candidate fails held-out, validate original gain=2/ramp=3 at 1.5 m/s on separate seeds 96000..96009",
        "gate": "same as g1lab.lab_evaluation.trial; unchanged thresholds",
        "diagnostics": "torque clipping across all 10 substeps; foot-body origin XY speed sampled at 50 Hz during ground contact is a slip proxy, not contact-point slip",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    rows = []
    started = time.perf_counter()

    def evaluate(speed, gain, ramp, seeds, hold, stage):
        group = []
        for seed in seeds:
            row, trace = trial(
                env, speed, seed, hold=hold, heading_hold=True, heading_gain=gain, ramp=ramp, diagnostics=True
            )
            row.update(stage=stage, heading_gain=gain, ramp_s=ramp)
            rows.append(row)
            group.append(row)
            write_csv(output / f"{stage}-v{speed:g}-g{gain:g}-r{ramp:g}-s{seed}.csv", trace)
            write_csv(output / "episodes.csv", rows)
            print(json.dumps(row), flush=True)
        return group

    candidates = []
    for speed, gain, ramp in itertools.product([1.5, 1.6, 1.65], [1.0, 2.0, 4.0], [3.0, 5.0]):
        group = evaluate(speed, gain, ramp, [94000, 94001], 30, "development")
        if all(row["passed"] for row in group):
            candidates.append(
                {
                    "speed": speed,
                    "gain": gain,
                    "ramp": ramp,
                    "yaw": float(np.mean([row["yaw_rate_rmse_rad_s"] for row in group])),
                }
            )
    selected = max(candidates, key=lambda c: (c["speed"], -c["yaw"])) if candidates else None
    validated = None
    if selected:
        group = evaluate(
            selected["speed"], selected["gain"], selected["ramp"], range(95000, 95010), 60, "validation"
        )
        if all(row["passed"] for row in group):
            validated = selected
    if validated is None:
        group = evaluate(1.5, 2.0, 3.0, range(96000, 96010), 60, "fallback_validation")
        if all(row["passed"] for row in group):
            validated = {"speed": 1.5, "gain": 2.0, "ramp": 3.0}
    result = {
        "status": "completed",
        "selected_development": selected,
        "validated": validated,
        "episodes": len(rows),
        "elapsed_wall_s": time.perf_counter() - started,
        "note": "Finite nominal-plane search; no global maximum or robustness claim; CPU physics/ONNX",
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
