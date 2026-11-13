"""Export the actor of an RSL-RL checkpoint into a policy bundle for benchmark_lab_policy.py.

Bundle layout: <output>/exported/policy.onnx (input "obs", output "actions", same as the public export)
and <output>/params/deploy.yaml copied from the training run, so PD gains and joint maps travel with it.
Only plain MLP actors without observation normalization are supported; anything else is rejected.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch


def main(args):
    checkpoint = Path(args.checkpoint).resolve()
    run = checkpoint.parent
    deploy = run / "params/deploy.yaml"
    if not deploy.is_file():
        raise FileNotFoundError(deploy)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
    if any("normalizer" in k for k in state):
        raise ValueError("Checkpoint uses observation normalization; this exporter does not fold it in")
    indices = sorted({int(k.split(".")[1]) for k in state if k.startswith("actor.")})
    layers = []
    for n, i in enumerate(indices):
        weight, bias = state[f"actor.{i}.weight"], state[f"actor.{i}.bias"]
        linear = torch.nn.Linear(weight.shape[1], weight.shape[0])
        linear.weight.data.copy_(weight)
        linear.bias.data.copy_(bias)
        layers.append(linear)
        if n < len(indices) - 1:
            layers.append(torch.nn.ELU())
    actor = torch.nn.Sequential(*layers).eval()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "exported").mkdir()
    (output / "params").mkdir()
    shutil.copy2(deploy, output / "params/deploy.yaml")
    onnx_path = output / "exported/policy.onnx"
    dummy = torch.zeros(1, actor[0].in_features)
    torch.onnx.export(
        actor, dummy, onnx_path, input_names=["obs"], output_names=["actions"], opset_version=11, dynamo=False
    )
    probe = torch.randn(4, actor[0].in_features, generator=torch.Generator().manual_seed(0))
    meta = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "iteration": torch.load(checkpoint, map_location="cpu", weights_only=False).get("iter"),
        "onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        "probe_actions_first_row": actor(probe)[0].tolist(),
    }
    (output / "export.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
