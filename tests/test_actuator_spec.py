import math

from g1lab.actuator_spec import SPEC, clip_torque, torque_limit


def test_spec_covers_all_29_joints_and_is_never_looser_than_any_source():
    assert len(SPEC) == 29
    mujoco_ctrl = {
        "hip_pitch": 88,
        "hip_roll": 88,
        "hip_yaw": 88,
        "knee": 139,
        "ankle_pitch": 50,
        "ankle_roll": 50,
    }
    for name, spec in SPEC.items():
        for key, cap in mujoco_ctrl.items():
            if f"_{key}_joint" in name:
                assert spec["y1"] <= cap and spec["y2"] <= cap
    assert SPEC["left_knee_joint"]["y1"] == 111.0 and SPEC["left_knee_joint"]["x2"] == 22.7
    assert SPEC["left_hip_roll_joint"]["y2"] == 88.0  # MuJoCo cap below the 131 N*m curve value


def test_torque_speed_curve_derates_linearly_to_zero_at_no_load_speed():
    knee = SPEC["left_knee_joint"]
    assert torque_limit(knee, 5.0, 1.0) == 111.0  # same direction, below knee point
    assert torque_limit(knee, 5.0, -1.0) == 131.0  # opposite direction (braking)
    mid = (knee["x1"] + knee["x2"]) / 2
    assert math.isclose(torque_limit(knee, mid, 1.0), 55.5)
    assert torque_limit(knee, 30.0, 1.0) == 0.0
    assert clip_torque(knee, 0.0, 500.0) == 131.0  # zero speed counts as opposite direction (Unitree model)
    ankle = SPEC["left_ankle_pitch_joint"]
    assert torque_limit(ankle, 100.0, 1.0) == 25.0  # flat limit, speed handled by the velocity limit
