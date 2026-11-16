"""Single G1 29-DoF actuator envelope used by both simulators ("do not exceed the technical spec").

Sources disagree, so each joint takes the most conservative value of:
* Unitree torque-speed test data in vendor unitree_rl_lab `unitree_actuators.py` (N7520-14.3, N7520-22.5):
  Y1 peak torque with torque and speed in the same direction, Y2 opposite direction, X1 knee-point speed
  (full torque below it), X2 no-load speed (torque falls linearly to zero at X2);
* effort/velocity limits of the pinned Isaac robot config (`unitree.py`, UNITREE_G1_29DOF_CFG);
* actuator ctrlrange of the Unitree MuJoCo model (`g1_29dof.xml`).
Joints without torque-speed data get a flat torque limit (Y1 = Y2, no derating) and the Isaac velocity limit.
Pure Python: importable without Isaac or MuJoCo.
"""

import math

INF = math.inf
_N7520_14 = {"y1": 71.0, "y2": 83.3, "x1": 22.63, "x2": 35.52, "fs": 1.6, "fd": 0.16, "motor": "N7520-14.3"}
_N7520_22 = {"y1": 111.0, "y2": 131.0, "x1": 14.5, "x2": 22.7, "fs": 2.4, "fd": 0.24, "motor": "N7520-22.5"}


def _flat(torque, velocity, motor):
    return {
        "y1": torque,
        "y2": torque,
        "x1": INF,
        "x2": INF,
        "vel": velocity,
        "fs": 0.0,
        "fd": 0.0,
        "motor": motor,
    }


def _curve(base, cap=None):
    spec = dict(base)
    if cap is not None:  # MuJoCo model gives hip roll only 88 N*m
        spec["y1"], spec["y2"] = min(spec["y1"], cap), min(spec["y2"], cap)
    spec["vel"] = spec["x2"]
    return spec


SPEC = {}
for side in ("left", "right"):
    SPEC[f"{side}_hip_pitch_joint"] = _curve(_N7520_14)
    SPEC[f"{side}_hip_yaw_joint"] = _curve(_N7520_14)
    SPEC[f"{side}_hip_roll_joint"] = _curve(_N7520_22, cap=88.0)
    SPEC[f"{side}_knee_joint"] = _curve(_N7520_22)
    SPEC[f"{side}_ankle_pitch_joint"] = _flat(25.0, 37.0, "N5020-16 (Isaac 25 < MuJoCo 50)")
    SPEC[f"{side}_ankle_roll_joint"] = _flat(25.0, 37.0, "N5020-16 (Isaac 25 < MuJoCo 50)")
    for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll"):
        SPEC[f"{side}_{joint}_joint"] = _flat(25.0, 37.0, "N5020-16")
    for joint in ("wrist_pitch", "wrist_yaw"):
        SPEC[f"{side}_{joint}_joint"] = _flat(5.0, 22.0, "W4010-25")
SPEC["waist_yaw_joint"] = _curve(_N7520_14)
SPEC["waist_roll_joint"] = _flat(25.0, 37.0, "N5020-16 (Isaac 25 < MuJoCo 50)")
SPEC["waist_pitch_joint"] = _flat(25.0, 37.0, "N5020-16 (Isaac 25 < MuJoCo 50)")


def torque_limit(spec, velocity, torque):
    """Admissible |torque| for a requested torque at joint velocity (Unitree actuator model semantics)."""
    peak = spec["y1"] if velocity * torque > 0 else spec["y2"]
    speed = abs(velocity)
    if speed <= spec["x1"]:
        return peak
    if spec["x2"] <= spec["x1"]:
        return 0.0
    return max(0.0, peak * (spec["x2"] - speed) / (spec["x2"] - spec["x1"]))


def clip_torque(spec, velocity, torque):
    limit = torque_limit(spec, velocity, torque)
    return max(-limit, min(limit, torque))
