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


def test_knobs_never_raise_heading_bonuses():
    for knob in autopilot.KNOBS:
        for key in knob.get("overrides", {}):
            assert "yaw" not in key and "ang_vel" not in key and "twist" not in key, (knob["name"], key)


def collapse_pilot(tmp_path):
    anchor = tmp_path / "anchor.pt"
    anchor.write_bytes(b"x")
    pilot = fake_pilot(tmp_path, {})
    pilot.args = SimpleNamespace(max_startup_failures=3, capability_drop=1.5, stall_rounds=3)
    pilot.state.update(
        {
            "round": 20,
            "level": 7.2,
            "latest_ckpt": str(tmp_path / "latest.pt"),
            "best_isaac": {"speed": 3.0, "vx": 3.0, "round": 18, "ckpt": str(tmp_path / "isaac.pt")},
            "best_capability": {"speed": 7.0, "round": 15},
            "best_mujoco": {"speed": 5.5, "round": 11, "validation_10seeds_60s_max_passed_mps": 5.5},
            "rounds": [{"round": 11, "train": {"checkpoint": str(anchor)}}],
            "consecutive_failures": 0,
        }
    )
    pilot.keep_capability = lambda *a: None
    return pilot, anchor


def ev(strict, cap):
    return {
        "max_sustained_speed_mps": strict,
        "max_speed_ignoring_yaw_mps": cap,
        "groups": [{"speed_cmd_mps": strict, "mean_vx_mps": strict}],
    }


def test_capability_collapse_rolls_back_to_mujoco_anchor(tmp_path):
    pilot, anchor = collapse_pilot(tmp_path)
    pilot.decide(21, str(tmp_path / "r21.pt"), ev(3.0, 5.5), None)  # first low round: keep going
    assert pilot.state["latest_ckpt"] == str(tmp_path / "r21.pt")
    pilot.decide(22, str(tmp_path / "r22.pt"), ev(3.0, 4.0), None)  # second in a row: collapse
    assert pilot.state["latest_ckpt"] == str(anchor)
    assert pilot.state["level"] == 6.0  # anchor speed + 0.5
    assert pilot.state["lr"] == 5e-4


def test_small_capability_dip_does_not_roll_back(tmp_path):
    pilot, _ = collapse_pilot(tmp_path)
    for n in (21, 22, 23):
        pilot.decide(n, str(tmp_path / f"r{n}.pt"), ev(3.0, 6.0), None)
    assert pilot.state["latest_ckpt"] == str(tmp_path / "r23.pt")


def test_zero_speed_round_rolls_back_to_validated_init(tmp_path):
    """Run v3-20261117: round 1 collapsed to 0 m/s and became 'best'; the anchor must stay the init policy."""
    init = tmp_path / "init.pt"
    init.write_bytes(b"x")
    pilot = fake_pilot(tmp_path, {})
    pilot.args = SimpleNamespace(max_startup_failures=3, capability_drop=1.5, stall_rounds=3)
    pilot.state.update(
        {
            "round": 1,
            "level": 5.5,
            "latest_ckpt": str(init),
            "init_ckpt": str(init),
            "init_speed": 5.5,
            "best_isaac": {"speed": 0.0, "vx": 0.0, "round": 1, "ckpt": str(tmp_path / "r1.pt")},
            "best_capability": {"speed": 0.0, "round": 1},
            "best_mujoco": {"speed": 0.0, "round": 1},
            "rounds": [{"round": 1, "train": {"checkpoint": str(tmp_path / "r1.pt")}}],
            "consecutive_failures": 0,
        }
    )
    pilot.keep_capability = lambda *a: None
    pilot.decide(2, str(tmp_path / "r2.pt"), ev(0.0, 0.0), None)
    assert pilot.state["latest_ckpt"] == str(init)
    assert pilot.state["level"] == 5.5  # min(level, 5.5 + 0.5)
