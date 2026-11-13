"""Prepare a reviewable speed-stage command; execute only on a configured RTX/Isaac host."""

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from g1lab.fast_plan import (
    EXPERIMENT_NAME,
    LAB_REVISION,
    SPRINT_EXPERIMENT,
    command,
    sprint_command,
    stage_settings,
)
from g1lab.gpu import pin_cuda
from g1lab.isaac_tasks import V0_GAINS_TASK


def provenance(source, gpu):
    """Versions and hardware needed to reproduce a run; collected without importing Isaac."""
    import importlib.metadata

    packages = {}
    for name in [
        "torch",
        "isaacsim",
        "isaaclab",
        "isaaclab_rl",
        "isaaclab_tasks",
        "rsl-rl-lib",
        "unitree_rl_lab",
    ]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    def git(path):
        try:
            return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).split()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        driver = None
    import torch

    return {
        "python": platform.python_version(),
        "packages": packages,
        "torch_cuda": torch.version.cuda,
        "gpu": gpu,
        "driver": driver,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "project_commit": git(Path(__file__).resolve().parents[1]),
        "project_dirty": bool(
            subprocess.run(
                ["git", "-C", str(Path(__file__).resolve().parents[1]), "diff", "--quiet", "HEAD"]
            ).returncode
        ),
        "unitree_rl_lab_commit": git(source),
    }


def main(args):
    pin_cuda()  # before torch is imported here or in the training subprocess
    source = Path(args.source).resolve()
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != LAB_REVISION:
        raise ValueError(
            "Upstream revision differs from the audited version; fetch the pinned revision first"
        )
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else None
    staged = None
    if checkpoint:
        if not checkpoint.is_file() or checkpoint.suffix != ".pt":
            raise ValueError("Supply an existing full RSL-RL .pt checkpoint, not ONNX")
        import hashlib

        key = hashlib.sha256(checkpoint.read_bytes()).hexdigest()[:12]
        experiment = SPRINT_EXPERIMENT if args.profile == "sprint" else EXPERIMENT_NAME
        staged = source / "logs/rsl_rl" / experiment / f"resume_{key}" / checkpoint.name
    if args.profile == "sprint":
        argv = sprint_command(
            source,
            args.python,
            args.usd,
            args.envs,
            args.iterations,
            args.seed,
            staged,
            critic_warmup=args.critic_warmup,
            extra=args.train_arg,
            learning_rate=args.learning_rate,
        )
    else:
        argv = command(
            source,
            args.python,
            args.usd,
            args.cap,
            args.envs,
            args.iterations,
            args.seed,
            staged,
            task=args.task,
            critic_warmup=args.critic_warmup,
        )
    metadata = {
        "argv": argv,
        "source_revision": revision,
        "settings": stage_settings(args.cap),
        "task": args.task if args.profile != "sprint" else "Unitree-G1-29dof-Sprint-G1Lab",
        "profile": args.profile,
        "critic_warmup": args.critic_warmup,
        "status": "prepared_not_executed",
        "checkpoint_source": str(checkpoint) if checkpoint else None,
        "checkpoint_stage": str(staged) if staged else None,
        "note": "Experimental stage configuration. No automatic stage promotion. Isaac runtime unvalidated on this machine.",
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "command.json").write_text(json.dumps(metadata, indent=2))
    if not args.execute:
        print(json.dumps(metadata, indent=2))
        return
    if Path(args.python).resolve() != Path(sys.executable).resolve():
        raise ValueError("Run this launcher using the same Isaac-enabled Python supplied by --python")
    missing = [name for name in ["isaaclab", "isaacsim"] if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f"Isaac environment missing: {missing}; command remains prepared only")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is unavailable in this interpreter")
    gpu = torch.cuda.get_device_name(0)
    if "GTX" in gpu.upper():
        raise RuntimeError("This Isaac Sim pipeline requires an RTX-capable GPU; GTX is unsupported")
    if not Path(args.usd).is_file():
        raise FileNotFoundError("Provide the actual official G1 29-DoF USD asset path")
    if staged:
        staged.parent.mkdir(parents=True, exist_ok=True)
        if not staged.exists():
            shutil.copy2(checkpoint, staged)
    metadata["status"] = "running"
    metadata["provenance"] = provenance(source, gpu)
    (output / "command.json").write_text(json.dumps(metadata, indent=2))
    log_root = source / "logs/rsl_rl" / (SPRINT_EXPERIMENT if args.profile == "sprint" else EXPERIMENT_NAME)
    before = set(log_root.iterdir()) if log_root.is_dir() else set()
    started = time.time()
    with (output / "train.log").open("w") as log:
        result = subprocess.run(argv, cwd=source, stdout=log, stderr=subprocess.STDOUT)
    metadata["wall_time_s"] = time.time() - started
    runs = sorted(set(log_root.iterdir()) - before) if log_root.is_dir() else []
    metadata["run_dirs"] = [str(run) for run in runs]
    metadata["status"] = "finished" if result.returncode == 0 else "failed"
    metadata["returncode"] = result.returncode
    (output / "command.json").write_text(json.dumps(metadata, indent=2))
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="vendor/unitree_rl_lab")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--usd", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cap", type=float, default=1.0)
    parser.add_argument("--envs", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--task", default=V0_GAINS_TASK, help="Isaac task id; default matches public ONNX PD gains"
    )
    parser.add_argument("--critic-warmup", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=None, help="sprint profile only")
    parser.add_argument("--profile", choices=["speed-stage", "sprint"], default="speed-stage")
    parser.add_argument(
        "--train-arg", action="append", default=[], help="extra train.py argument (repeatable)"
    )
    parser.add_argument("--execute", action="store_true")
    main(parser.parse_args())
