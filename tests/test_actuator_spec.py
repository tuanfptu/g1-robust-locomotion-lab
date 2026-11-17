import math
import re
from pathlib import Path

import pytest

from g1lab.actuator_spec import MUJOCO_PEAK, SPEC, clip_torque, torque_limit

XML = Path("vendor/unitree_mujoco/unitree_robots/g1/g1_29dof.xml")


def test_spec_covers_all_29_joints_with_mujoco_peak_torques():
    assert len(SPEC) == 29
    for name, spec in SPEC.items():
        key = re.sub(r"^(left|right)_|_joint$", "", name)
        assert spec["y1"] == spec["y2"] == MUJOCO_PEAK[key]


@pytest.mark.skipif(not XML.is_file(), reason="vendor MuJoCo model not installed")
def test_peaks_match_the_mujoco_model_ctrlrange():
    ctrl = dict(re.findall(r'<motor name="([a-z_]+)"[^>]*ctrlrange="-?[\d.]+ ([\d.]+)"', XML.read_text()))
    assert ctrl, "no motors parsed"
    for name, spec in SPEC.items():
        assert spec["y1"] == float(ctrl[name.removesuffix("_joint")])


def test_torque_speed_curve_derates_linearly_to_zero_at_no_load_speed():
    knee = SPEC["left_knee_joint"]
    assert torque_limit(knee, 5.0, 1.0) == 139.0
    mid = (knee["x1"] + knee["x2"]) / 2
    assert math.isclose(torque_limit(knee, mid, 1.0), 69.5)
    assert torque_limit(knee, 30.0, 1.0) == 0.0
    assert clip_torque(knee, 0.0, 500.0) == 139.0
    assert torque_limit(SPEC["left_ankle_pitch_joint"], 100.0, 1.0) == 50.0
