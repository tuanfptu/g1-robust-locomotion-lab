"""Prepare a reviewable speed-stage command; execute only on a configured RTX/Isaac host."""

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

from g1lab.fast_plan import LAB_REVISION, command, stage_settings


def main(args):
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
        staged = source / "logs/rsl_rl/g1_fast" / f"resume_{key}" / checkpoint.name
    argv = command(source, args.python, args.usd, args.cap, args.envs, args.iterations, args.seed, staged)
    metadata = {
        "argv": argv,
        "source_revision": revision,
        "settings": stage_settings(args.cap),
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
    (output / "command.json").write_text(json.dumps(metadata, indent=2))
    with (output / "train.log").open("w") as log:
        result = subprocess.run(argv, cwd=source, stdout=log, stderr=subprocess.STDOUT)
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
    parser.add_argument("--execute", action="store_true")
    main(parser.parse_args())
