"""Six-task PPO with behavioral replay from the previous competent student policy."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from mixed_campaign import execute

from g1lab.config import EnvConfig
from g1lab.selection import acceptable
from g1lab.training import load_policy, policy_action

TASKS = ("flat", "slope", "rough", "stand", "lateral", "turn")


def validate(bundle, directory, seed, episodes, seconds):
    directory.mkdir()
    result = {}
    for task in TASKS:
        output = directory / task
        execute(
            [
                "eval",
                "--bundle",
                str(bundle),
                "--config",
                f"configs/mixed/{task}.json",
                "--output",
                str(output),
                "--seed",
                str(seed),
                "--episodes",
                str(episodes),
                "--seconds",
                str(seconds),
            ],
            directory / f"{task}.log",
        )
        result[task] = json.loads((output / "summary.json").read_text())["configured"]
    return result


def main(args):
    torch.set_num_threads(1)
    root = Path(args.run)
    root.mkdir(parents=True, exist_ok=False)
    (root / "protocol.json").write_text(
        json.dumps(
            {
                **vars(args),
                "tasks": TASKS,
                "method": "PPO with replay labels from previous student; not new expert labels",
                "validation_seed": 7000,
                "heldout_seeds": [50000, 50001, 50002],
            },
            indent=2,
        )
    )
    observations, actions = [], []
    for i, task in enumerate(TASKS):
        config = EnvConfig(**json.loads(Path(f"configs/mixed/{task}.json").read_text()))
        model, vec, raw = load_policy(args.bundle, config)
        try:
            obs, _ = raw.reset(seed=100 + i)
            for _ in range(1536):
                action = policy_action(model, vec, obs)
                observations.append(obs.copy())
                actions.append(action.copy())
                obs, _, done, truncated, _ = raw.step(action)
                if done or truncated:
                    obs, _ = raw.reset()
        finally:
            vec.close()
    anchor = root / "skill-replay.npz"
    np.savez_compressed(anchor, observations=np.array(observations), actions=np.array(actions))
    print("Collected 9216 previous-policy replay labels", flush=True)
    baseline = validate(
        args.bundle, root / "baseline-validation", 7000, args.validation_episodes, args.validation_seconds
    )
    best, reference = args.bundle, baseline
    progress = []
    mix = [arg for task in TASKS[1:] for arg in ("--mix-config", f"configs/mixed/{task}.json")]
    for iteration in range(args.rounds):
        run = root / f"round-{iteration}"
        execute(
            [
                "train",
                "--config",
                "configs/mixed/flat.json",
                *mix,
                "--run",
                str(run),
                "--resume",
                best,
                "--steps",
                "18432",
                "--envs",
                "6",
                "--workers",
                "--threads",
                "1",
                "--learning-rate",
                "0.000005",
                "--target-kl",
                "0.01",
                "--freeze-actor-steps",
                "6144",
                "--anchor-data",
                str(anchor),
                "--freeze-normalization",
                "--checkpoint-every",
                "12288",
            ],
            root / f"round-{iteration}.log",
        )
        candidate = str(run / "final")
        metrics = validate(
            candidate,
            root / f"validation-{iteration}",
            7000,
            args.validation_episodes,
            args.validation_seconds,
        )
        accepted = acceptable(metrics, reference, yaw_tolerance=0.05)
        progress.append({"round": iteration, "accepted": accepted, "metrics": metrics})
        if accepted:
            best, reference = candidate, metrics
        (root / "progress.json").write_text(json.dumps(progress, indent=2))
        print(json.dumps({"round": iteration, "accepted": accepted, "selected": best}), flush=True)
    shutil.copytree(best, root / "selected")
    (root / "selection.json").write_text(json.dumps({"source": best, "metrics": reference}, indent=2))
    heldout = validate(root / "selected", root / "heldout", 50000, 3, 30)
    (root / "heldout.json").write_text(json.dumps(heldout, indent=2))
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--validation-episodes", type=int, default=2)
    parser.add_argument("--validation-seconds", type=float, default=30)
    main(parser.parse_args())
