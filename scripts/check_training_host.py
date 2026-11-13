"""Collect portable preflight evidence; this does not launch or certify Isaac Sim."""

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def collect(usd=None):
    packages = {}
    for name in ["torch", "mujoco", "onnxruntime", "isaacsim", "isaaclab", "isaaclab-rl", "rsl-rl-lib"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        gpu = (result.stdout or result.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        gpu = str(error)
    import torch

    cuda = {
        "available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "tensor_smoke_passed": False,
    }
    if cuda["available"]:
        try:
            cuda["device"] = torch.cuda.get_device_name(0)
            cuda["capability"] = list(torch.cuda.get_device_capability(0))
            cuda["compiled_architectures"] = torch.cuda.get_arch_list()
            matrix = torch.eye(32, device="cuda")
            cuda["tensor_smoke_passed"] = bool(torch.allclose(matrix @ matrix, matrix))
            torch.cuda.synchronize()
        except Exception as error:
            cuda["error"] = str(error)
    missing = [
        name
        for name in ["isaaclab", "isaacsim", "isaaclab_rl", "rsl_rl", "unitree_rl_lab"]
        if importlib.util.find_spec(name) is None
    ]
    blockers = []
    if not cuda["tensor_smoke_passed"]:
        blockers.append("CUDA tensor smoke failed/unavailable in this Python")
    if "GTX" in gpu.upper():
        blockers.append("GTX GPU does not support this Isaac Sim pipeline")
    if missing:
        blockers.append(f"Isaac imports not installed: {missing}")
    if not usd or not Path(usd).is_file():
        blockers.append("Actual G1 USD asset path missing")
    usage = shutil.disk_usage(Path.cwd())
    try:
        memory = next(
            line for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemTotal:")
        )
    except (OSError, StopIteration):
        memory = "unknown"
    return {
        "python": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "memory": memory,
        "disk_free_gib": usage.free / 2**30,
        "packages": packages,
        "nvidia_smi": gpu,
        "cuda": cuda,
        "usd": str(usd) if usd else None,
        "blockers": blockers,
        "preflight_passed": not blockers,
        "isaac_runtime_smoke_tested": False,
        "note": "Even a passing preflight requires an Isaac 32-env short training smoke before scaling.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--usd")
    parser.add_argument("--require-training", action="store_true")
    args = parser.parse_args()
    report = collect(args.usd)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if args.require_training and not report["preflight_passed"]:
        raise SystemExit(1)
