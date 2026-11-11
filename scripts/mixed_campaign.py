"""Local interleaved PPO+BC pilot with a three-terrain validation gate."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from g1lab.selection import acceptable, score


def execute(command, log):
    with log.open("w") as handle:
        subprocess.run(
            [sys.executable, "-m", "g1lab.cli", *command], stdout=handle, stderr=subprocess.STDOUT, check=True
        )


def validate(bundle, directory, seed, episodes=2, seconds=15):
    result = {}
    for terrain in ("flat", "slope", "rough"):
        output = directory / terrain
        execute(
            [
                "eval",
                "--bundle",
                str(bundle),
                "--config",
                f"configs/mixed/{terrain}.json",
                "--output",
                str(output),
                "--seed",
                str(seed),
                "--episodes",
                str(episodes),
                "--seconds",
                str(seconds),
            ],
            directory / f"{terrain}.log",
        )
        result[terrain] = json.loads((output / "summary.json").read_text())["configured"]
    return result


def main(args):
    root = Path(args.run)
    root.mkdir(parents=True, exist_ok=False)
    (root / "protocol.json").write_text(json.dumps(vars(args), indent=2))
    sources = [
        Path(args.bundle) / "policy.zip",
        Path(args.bundle) / "vecnormalize.pkl",
        *[Path(path) for path in args.data],
    ]
    (root / "input-sha256.json").write_text(
        json.dumps({str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}, indent=2)
    )
    rng = np.random.default_rng(42)
    observations, actions = [], []
    for source in args.data:
        with np.load(source) as data:
            indices = rng.choice(
                len(data["observations"]), size=min(16000, len(data["observations"])), replace=False
            )
            observations.append(data["observations"][indices])
            actions.append(data["actions"][indices])
    anchor = root / "replay.npz"
    np.savez_compressed(anchor, observations=np.concatenate(observations), actions=np.concatenate(actions))
    best = args.bundle
    baseline_dir = root / "baseline-validation"
    baseline_dir.mkdir()
    reference = validate(best, baseline_dir, 6000)
    results = [{"stage": "baseline", "bundle": best, "metrics": reference}]
    for iteration in range(args.rounds):
        run = root / f"round-{iteration}"
        execute(
            [
                "train",
                "--config",
                "configs/mixed/flat.json",
                "--mix-config",
                "configs/mixed/slope.json",
                "--mix-config",
                "configs/mixed/rough.json",
                "--run",
                str(run),
                "--resume",
                best,
                "--steps",
                str(args.steps),
                "--envs",
                "3",
                "--workers",
                "--threads",
                "1",
                "--learning-rate",
                "0.000005",
                "--target-kl",
                "0.01",
                "--freeze-actor-steps",
                "1536",
                "--anchor-data",
                str(anchor),
                "--freeze-normalization",
                "--checkpoint-every",
                "8192",
            ],
            root / f"round-{iteration}.log",
        )
        directory = root / f"validation-{iteration}"
        directory.mkdir()
        candidate = str(run / "final")
        metrics = validate(candidate, directory, 6000)
        accepted = acceptable(metrics, reference)
        results.append(
            {
                "stage": iteration,
                "bundle": candidate,
                "accepted": accepted,
                "metrics": metrics,
                "score": score(metrics),
            }
        )
        if accepted:
            best, reference = candidate, metrics
        (root / "progress.json").write_text(json.dumps(results, indent=2))
        print(
            json.dumps({"round": iteration, "accepted": accepted, "selected": best, "score": score(metrics)}),
            flush=True,
        )
    shutil.copytree(best, root / "selected")
    (root / "selection.json").write_text(
        json.dumps({"source": best, "validation_seed": 6000, "metrics": reference}, indent=2)
    )
    # Fresh seeds are used only after selection; never used by the acceptance gate.
    final_dir = root / "heldout"
    final_dir.mkdir()
    heldout = validate(root / "selected", final_dir, 40000, episodes=3, seconds=30)
    (root / "heldout.json").write_text(json.dumps(heldout, indent=2))
    baseline_test = root / "baseline-heldout"
    baseline_test.mkdir()
    baseline = validate(args.bundle, baseline_test, 40000, episodes=3, seconds=30)
    (root / "baseline-heldout.json").write_text(json.dumps(baseline, indent=2))
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--data", action="append", required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=16384)
    main(parser.parse_args())
