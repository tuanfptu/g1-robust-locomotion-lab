"""Build an RSL-RL resume checkpoint whose actor is copied exactly from a public ONNX export.

The ONNX file only holds the actor mean network. The critic is freshly initialised (seeded), the action
std is set explicitly, the optimizer state is empty and the iteration counter is 0. Train with
`isaac_train.py --critic_warmup N` so the value function is fitted before actor updates.

A template checkpoint from the same task/agent config (e.g. a 2-iteration smoke run) supplies parameter
names, shapes and optimizer parameter groups; its trained values are not reused.
"""

import argparse
import hashlib
import json
from pathlib import Path

import onnx
import torch
from onnx import numpy_helper


def main(args):
    template = torch.load(args.template, map_location="cpu", weights_only=False)
    state = template["model_state_dict"]
    onnx_weights = {
        t.name: torch.from_numpy(numpy_helper.to_array(t).copy())
        for t in onnx.load(args.onnx).graph.initializer
    }
    actor_keys = sorted(k for k in state if k.startswith("actor."))
    if sorted(onnx_weights) != actor_keys:
        raise ValueError(f"ONNX parameters {sorted(onnx_weights)} do not match actor {actor_keys}")
    if any(k.endswith("normalizer") or "normalizer" in k for k in state):
        raise ValueError("Template uses observation normalization; the ONNX export has none")
    new_state = {}
    generator = torch.Generator().manual_seed(args.seed)
    for key, value in state.items():
        if key.startswith("actor."):
            if onnx_weights[key].shape != value.shape:
                raise ValueError(
                    f"{key}: ONNX {tuple(onnx_weights[key].shape)} vs actor {tuple(value.shape)}"
                )
            new_state[key] = onnx_weights[key].to(value.dtype)
        elif key.startswith("critic."):
            # Same rule as torch.nn.Linear.reset_parameters: U(-1/sqrt(fan_in), 1/sqrt(fan_in)).
            layer = key.rsplit(".", 1)[0]
            fan_in = state[f"{layer}.weight"].shape[1]
            bound = 1.0 / fan_in**0.5
            new_state[key] = (torch.rand(value.shape, generator=generator) * 2 - 1) * bound
        elif key == "std":
            new_state[key] = torch.full_like(value, args.std)
        elif key == "log_std":
            new_state[key] = torch.full_like(value, float(torch.log(torch.tensor(args.std))))
        else:
            raise ValueError(f"Unexpected parameter {key}; refusing to guess")
    optimizer = template["optimizer_state_dict"]
    groups = [dict(group, lr=args.learning_rate) for group in optimizer["param_groups"]]
    onnx_sha = hashlib.sha256(Path(args.onnx).read_bytes()).hexdigest()
    infos = {
        "g1lab_init": "onnx_actor_transplant",
        "onnx": str(Path(args.onnx).resolve()),
        "onnx_sha256": onnx_sha,
        "template": str(Path(args.template).resolve()),
        "std": args.std,
        "critic_seed": args.seed,
        "note": "Critic untrained; actor exact copy of ONNX. Not a full checkpoint of the original run.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    torch.save(
        {
            "model_state_dict": new_state,
            "optimizer_state_dict": {"state": {}, "param_groups": groups},
            "iter": 0,
            "infos": infos,
        },
        output,
    )
    output.with_suffix(".json").write_text(json.dumps(infos, indent=2))
    print(json.dumps(infos, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--std", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    main(parser.parse_args())
