"""Budgeted PPO fine-tuning. Reject regressions using a separate validation seed."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main(args):
    root = Path(args.run)
    root.mkdir(parents=True, exist_ok=False)
    best = args.bundle
    results = []
    stages = ["commands", "robust", "slope", "rough"]
    extra = ["--anchor-data", args.anchor_data, "--freeze-normalization"] if args.anchor_data else []
    for stage in stages:
        run = root / stage
        with (root / f"{stage}.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "g1lab.cli",
                    "train",
                    "--run",
                    str(run),
                    "--config",
                    f"configs/bootstrapped-{stage}.json",
                    "--resume",
                    best,
                    "--steps",
                    str(args.steps),
                    "--envs",
                    "2",
                    "--workers",
                    "--learning-rate",
                    "0.00001",
                    "--target-kl",
                    "0.01",
                    "--freeze-actor-steps",
                    "4096",
                    "--checkpoint-every",
                    "8192",
                    *extra,
                ],
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        evaluation = root / f"{stage}-validation"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "g1lab.cli",
                "eval",
                "--bundle",
                str(run / "final"),
                "--config",
                "configs/bootstrapped-walk.json",
                "--output",
                str(evaluation),
                "--seed",
                "5000",
                "--episodes",
                "1",
                "--seconds",
                "10",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        metrics = json.loads((evaluation / "summary.json").read_text())["configured"]
        accepted = metrics["survival_rate"] == 1 and metrics["velocity_rmse_mps"]["mean"] < 0.15
        if accepted:
            best = str(run / "final")
        row = {
            "stage": stage,
            "accepted": accepted,
            "acceptance_scope": "flat-walk regression guard only, not terrain certification",
            "validation": metrics,
            "selected_bundle": best,
        }
        results.append(row)
        (root / "progress.json").write_text(json.dumps(results, indent=2))
        (root / "selected.txt").write_text(best + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--steps", type=int, default=16384)
    parser.add_argument("--anchor-data")
    main(parser.parse_args())
