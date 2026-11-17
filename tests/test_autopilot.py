"""Autopilot knobs must only touch reward terms that exist (a missing key made every launch fail on 2026-11-17)."""

import ast
import importlib.util
import json
import re
import threading
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("autopilot_sprint", ROOT / "scripts/autopilot_sprint.py")
autopilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(autopilot)


def reward_terms():
    tree = ast.parse((ROOT / "src/g1lab/isaac_sprint.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "SPRINT_WEIGHTS":
            names = {k.value for k in node.value.keys}
    vendor = (
        ROOT / "vendor/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof"
        "/velocity_env_cfg.py"
    )
    if vendor.exists():
        names |= set(re.findall(r"^\s{4}(\w+) = RewTerm\(", vendor.read_text(), re.MULTILINE))
    else:
        names |= {"track_ang_vel_z"}
    return names


def test_knob_reward_keys_exist():
    terms = reward_terms()
    for knob in autopilot.KNOBS:
        for key in knob.get("overrides", {}):
            if key.startswith("env.rewards."):
                assert key.split(".")[2] in terms, (knob["name"], key)


def fake_pilot(tmp_path, overrides):
    pilot = object.__new__(autopilot.Autopilot)
    pilot.args = SimpleNamespace(max_startup_failures=3)
    pilot.out = tmp_path
    pilot.state_path = tmp_path / "state.json"
    pilot.stop = threading.Event()
    pilot.state = {"overrides": dict(overrides), "lr": 1e-3, "knobs_applied": [], "stall": 0, "events": []}
    return pilot


def test_rejected_override_reverts_knob(tmp_path):
    pilot = fake_pilot(tmp_path, {"env.rewards.flight_phase.weight": 0.7})
    pilot.apply_knob()  # first knob
    first = autopilot.KNOBS[0]
    for key, value in first["overrides"].items():
        assert pilot.state["overrides"][key] == value
    round_dir = tmp_path / "round_001"
    (round_dir / "train").mkdir(parents=True)
    key = next(iter(first["overrides"]))
    (round_dir / "train/train.log").write_text(f"Could not override '{key}'.\nKey not in struct\n")
    assert pilot.startup_failure(1, round_dir)
    assert key not in pilot.state["overrides"]
    assert pilot.state["overrides"]["env.rewards.flight_phase.weight"] == 0.7
    assert any(e["kind"] == "knob_reverted" for e in pilot.state["events"])


def test_repeated_startup_failure_halts(tmp_path):
    pilot = fake_pilot(tmp_path, {})
    pilot.stop.set()  # makes the back-off wait return immediately
    round_dir = tmp_path / "round_001"
    (round_dir / "train").mkdir(parents=True)
    (round_dir / "train/train.log").write_text("Segmentation fault\n")
    results = [pilot.startup_failure(i, round_dir) for i in range(3)]
    assert results == [True, True, False]
    assert json.dumps(pilot.state["events"][-1]).count("repeated_startup_failures") == 1
