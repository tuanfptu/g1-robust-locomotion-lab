"""Aggregate expert labels on student-visited states; imitation, not from-scratch RL."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from g1lab.bootstrap import UnitreeTeacher
from g1lab.config import EnvConfig
from g1lab.training import load_policy, policy_action, save_bundle


def main(args):
    torch.set_num_threads(2)
    rng = np.random.default_rng(42)
    output = Path(args.run)
    output.mkdir(parents=True, exist_ok=False)
    config = EnvConfig(**json.loads((Path(args.bundle) / "env_config.json").read_text()))
    if args.terrain is not None:
        config = replace(config, terrain=args.terrain, difficulty=args.difficulty)
    model, vec, raw = load_policy(
        args.bundle,
        replace(config, random_commands=False, command=(0.3, 0, 0), domain_randomization=False, pushes=False),
    )
    with np.load(args.data) as data:
        observations, actions = data["observations"].copy(), data["actions"].copy()
    teacher = UnitreeTeacher(args.source)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=0.0001)
    results = []
    try:
        for iteration in range(args.rounds):
            beta = max(0, 0.8 - iteration * 0.2)
            obs, _ = raw.reset(seed=42 + iteration)
            teacher.reset()
            new_obs, new_actions = [], []
            for _ in range(args.samples):
                teacher.previous = raw.previous_action[:12] * config.action_scale / 0.25
                label = teacher.action(raw)
                predicted = policy_action(model, vec, obs)
                new_obs.append(obs.copy())
                new_actions.append(label.copy())
                action = beta * label + (1 - beta) * predicted
                obs, _, done, truncated, _ = raw.step(action)
                if done or truncated:
                    obs, _ = raw.reset()
                    teacher.reset()
            observations = np.concatenate([observations, np.array(new_obs)])
            actions = np.concatenate([actions, np.array(new_actions)])
            inputs = torch.from_numpy(vec.normalize_obs(observations).astype(np.float32))
            targets = torch.from_numpy(actions)
            for _ in range(args.epochs):
                for indices in np.array_split(rng.permutation(len(inputs)), max(1, len(inputs) // 256)):
                    predicted = model.policy.get_distribution(inputs[indices]).distribution.mean
                    loss = ((predicted[:, :12] - targets[indices, :12]) ** 2).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            # Validation seed is kept separate from final evaluation seeds 10000+.
            obs, _ = raw.reset(seed=5000)
            errors = []
            for step in range(500):
                obs, _, done, truncated, info = raw.step(policy_action(model, vec, obs))
                errors.append(info["velocity_error_sq"])
                if done or truncated:
                    break
            row = dict(
                round=iteration,
                beta=beta,
                validation_duration=(step + 1) * 0.02,
                validation_rmse=float(np.sqrt(np.mean(errors))),
                x_progress=info["x_progress_m"],
            )
            results.append(row)
            save_bundle(model, vec, output / f"round-{iteration}")
            (output / "progress.json").write_text(json.dumps(results, indent=2))
            print(json.dumps(row), flush=True)
        np.savez_compressed(output / "aggregated-data.npz", observations=observations, actions=actions)
    finally:
        vec.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--source", default="vendor/unitree_rl_gym")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--terrain", choices=("flat", "slope", "rough"))
    parser.add_argument("--difficulty", type=float, default=0.2)
    main(parser.parse_args())
