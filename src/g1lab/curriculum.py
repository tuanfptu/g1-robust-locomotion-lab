"""Explicit staged curriculum; stage transitions are budget-based, not a success claim."""

import json
import subprocess
import sys
from pathlib import Path

from .config import EnvConfig


def curriculum(args):
    root = Path(args.run)
    root.mkdir(parents=True, exist_ok=False)
    stages = [
        ("flat", 0, False),
        ("flat", 0, True),
        ("slope", 0.2, True),
        ("slope", 0.4, True),
        ("rough", 0.2, True),
        ("rough", 0.4, True),
    ]
    previous = args.resume
    for index, (terrain, difficulty, robust) in enumerate(stages):
        config = EnvConfig(terrain=terrain, difficulty=difficulty, domain_randomization=robust, pushes=robust)
        path = root / f"stage-{index}-{terrain}.json"
        path.write_text(json.dumps(config.to_dict(), indent=2))
        run = root / f"stage-{index}-{terrain}"
        command = [
            sys.executable,
            "-m",
            "g1lab.cli",
            "train",
            "--config",
            str(path),
            "--run",
            str(run),
            "--steps",
            str(args.steps_per_stage),
            "--envs",
            str(args.envs),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
        ]
        if previous:
            command += ["--resume", str(previous)]
        subprocess.run(command, check=True)
        # An interrupted stage must not silently start the next stage.
        if json.loads((run / "training.json").read_text())["interrupted"]:
            print(f"Stopped after interrupted stage: {run}")
            return
        previous = run / "final"
    print(f"Completed staged training. Evaluate all terrains to detect forgetting: {previous}")
