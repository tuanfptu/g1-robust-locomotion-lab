"""Command-line entry point. GPU use is opt-in; simulation stays on CPU."""

import argparse
import importlib.metadata
import json
import subprocess
import time


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Must be at least 1")
    return value


def positive_float(value):
    value = float(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("Must be positive")
    return value


def doctor(_):
    import torch

    from .env import G1Env

    versions = {
        name: importlib.metadata.version(name)
        for name in ("mujoco", "torch", "gymnasium", "stable-baselines3")
    }
    try:
        nvidia = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        gpu_status = (nvidia.stdout + nvidia.stderr).strip()
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        gpu_status = str(exc)
    env = G1Env()
    try:
        obs, _ = env.reset(seed=42)
        print(
            json.dumps(
                {
                    "versions": versions,
                    "cuda_available": torch.cuda.is_available(),
                    "gpu_status": gpu_status,
                    "joints": env.model.nu,
                    "observation_shape": list(obs.shape),
                    "backend": "CPU MuJoCo",
                },
                indent=2,
            )
        )
    finally:
        env.close()


def benchmark(args):
    import numpy as np

    from .config import EnvConfig
    from .env import G1Env

    env = G1Env(EnvConfig(terrain=args.terrain, difficulty=args.difficulty))
    start = time.perf_counter()
    resets = 0
    try:
        env.reset(seed=42)
        for _ in range(args.steps):
            _, _, terminated, truncated, _ = env.step(np.zeros(env.model.nu))
            if terminated or truncated:
                env.reset()
                resets += 1
    finally:
        env.close()
    elapsed = time.perf_counter() - start
    print(
        json.dumps(
            {
                "steps": args.steps,
                "seconds": elapsed,
                "control_steps_per_second": args.steps / elapsed,
                "resets": resets,
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description="G1 simulation-only RL lab for GTX 1660 Ti development")
    commands = parser.add_subparsers(dest="operation", required=True)
    commands.add_parser("doctor")
    bootstrap = commands.add_parser("bootstrap")
    bootstrap.add_argument("--run", required=True)
    bootstrap.add_argument("--source", default="vendor/unitree_rl_gym")
    bootstrap.add_argument("--samples", type=positive, default=12000)
    bootstrap.add_argument("--epochs", type=positive, default=60)
    bootstrap.add_argument("--seed", type=int, default=42)
    bootstrap.add_argument("--data", help="Reuse an existing teacher-data.npz dataset")
    train = commands.add_parser("train")
    train.add_argument("--config", default="configs/flat.json")
    train.add_argument("--run", required=True)
    train.add_argument("--steps", type=positive, default=1_000_000)
    train.add_argument("--envs", type=positive, default=1)
    train.add_argument("--rollout", type=positive, default=512)
    train.add_argument("--threads", type=positive, default=2)
    train.add_argument("--workers", action="store_true")
    train.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--checkpoint-every", type=positive, default=50_000)
    train.add_argument("--resume", help="Paired checkpoint directory from a previous run")
    train.add_argument(
        "--log-std-init",
        type=float,
        default=-1.5,
        help="Initial policy exploration log standard deviation (new policies only)",
    )
    train.add_argument("--learning-rate", type=float, default=0.0001)
    train.add_argument("--target-kl", type=float, default=0.03)
    train.add_argument("--anchor-data", help="Teacher-labelled data for explicit behavior-cloning anchoring")
    train.add_argument("--freeze-normalization", action="store_true")
    train.add_argument(
        "--freeze-actor-steps",
        type=int,
        default=0,
        help="Warm up the critic before updating an imitation-initialized actor",
    )
    curriculum = commands.add_parser("curriculum")
    curriculum.add_argument("--run", required=True)
    curriculum.add_argument("--steps-per-stage", type=positive, default=1_000_000)
    curriculum.add_argument("--envs", type=positive, default=1)
    curriculum.add_argument("--seed", type=int, default=42)
    curriculum.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    curriculum.add_argument("--resume")
    evaluate = commands.add_parser("eval")
    evaluate.add_argument("--bundle", help="Omit for an untrained nominal-PD baseline")
    evaluate.add_argument(
        "--teacher", help="Unitree source directory; explicitly evaluates pretrained teacher"
    )
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--suite", choices=("flat", "full"), default="flat")
    evaluate.add_argument("--episodes", type=positive, default=10)
    evaluate.add_argument("--seconds", type=positive_float, default=30)
    evaluate.add_argument("--seed", type=int, default=10000)
    evaluate.add_argument("--video", action="store_true")
    evaluate.add_argument("--config", help="Evaluate a single fixed-command environment configuration")
    evaluate.add_argument("--scenario", help="Evaluate only one named scenario from the suite")
    demo = commands.add_parser("demo")
    demo.add_argument("--bundle")
    demo.add_argument("--teacher", help="Unitree source directory; show pretrained teacher on project G1")
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument(
        "--auto", action="store_true", help="Automatically cycle stand, walk, gentle turn, stop"
    )
    bench = commands.add_parser("benchmark")
    bench.add_argument("--steps", type=positive, default=1000)
    for command in (demo, bench):
        command.add_argument("--terrain", choices=("flat", "slope", "rough", "stairs"), default="flat")
        command.add_argument("--difficulty", type=float, default=0.3)
    report = commands.add_parser("report")
    report.add_argument("inputs", nargs="+")
    report.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.operation == "doctor":
        doctor(args)
    elif args.operation == "benchmark":
        benchmark(args)
    elif args.operation == "bootstrap":
        from .bootstrap import bootstrap as execute

        execute(args)
    elif args.operation == "train":
        if args.rollout * args.envs < 2:
            parser.error("rollout * envs must be >= 2")
        from .training import train as execute

        execute(args)
    elif args.operation == "eval":
        from .evaluation import evaluate as execute

        execute(args)
    elif args.operation == "curriculum":
        from .curriculum import curriculum as execute

        execute(args)
    elif args.operation == "demo":
        from .demo import demo as execute

        execute(args)
    else:
        from .report import report as execute

        execute(args)


if __name__ == "__main__":
    main()
