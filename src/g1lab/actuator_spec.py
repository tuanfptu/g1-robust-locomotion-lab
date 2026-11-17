"""Single G1 29-DoF actuator envelope used by both simulators.

Peak torques follow the Unitree MuJoCo model (`g1_29dof.xml` actuator ctrlrange), as chosen by the project owner:
hip pitch/roll/yaw and waist yaw 88 N*m, knee 139, ankles and waist roll/pitch 50, shoulders/elbows/wrist roll 25,
wrist pitch/yaw 5. Speed limits are not given by that model, so the "do not exceed the technical spec" part keeps
Unitree's measured motor data (vendor unitree_rl_lab `unitree_actuators.py`) for the N7520 motors: torque is full up
to the knee-point speed X1 and falls linearly to zero at the no-load speed X2, plus motor friction. Other joints use
the velocity limits of the pinned Isaac robot config. A joint must never run faster than its rated speed.
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


MUJOCO_PEAK = {
    "hip_pitch": 88.0,
    "hip_roll": 88.0,
    "hip_yaw": 88.0,
    "knee": 139.0,
    "ankle_pitch": 50.0,
    "ankle_roll": 50.0,
    "waist_yaw": 88.0,
    "waist_roll": 50.0,
    "waist_pitch": 50.0,
    "shoulder_pitch": 25.0,
    "shoulder_roll": 25.0,
    "shoulder_yaw": 25.0,
    "elbow": 25.0,
    "wrist_roll": 25.0,
    "wrist_pitch": 5.0,
    "wrist_yaw": 5.0,
}


def _curve(base, peak):
    """Unitree speed derating (X1, X2, friction) with the MuJoCo-model peak torque in both directions."""
    spec = dict(base)
    spec["y1"] = spec["y2"] = peak
    spec["vel"] = spec["x2"]
    return spec


SPEC = {}
for side in ("left", "right"):
    SPEC[f"{side}_hip_pitch_joint"] = _curve(_N7520_14, MUJOCO_PEAK["hip_pitch"])
    SPEC[f"{side}_hip_yaw_joint"] = _curve(_N7520_14, MUJOCO_PEAK["hip_yaw"])
    SPEC[f"{side}_hip_roll_joint"] = _curve(_N7520_22, MUJOCO_PEAK["hip_roll"])
    SPEC[f"{side}_knee_joint"] = _curve(_N7520_22, MUJOCO_PEAK["knee"])
    for joint in ("ankle_pitch", "ankle_roll"):
        SPEC[f"{side}_{joint}_joint"] = _flat(MUJOCO_PEAK[joint], 37.0, "N5020-16")
    for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll"):
        SPEC[f"{side}_{joint}_joint"] = _flat(MUJOCO_PEAK[joint], 37.0, "N5020-16")
    for joint in ("wrist_pitch", "wrist_yaw"):
        SPEC[f"{side}_{joint}_joint"] = _flat(MUJOCO_PEAK[joint], 22.0, "W4010-25")
SPEC["waist_yaw_joint"] = _curve(_N7520_14, MUJOCO_PEAK["waist_yaw"])
SPEC["waist_roll_joint"] = _flat(MUJOCO_PEAK["waist_roll"], 37.0, "N5020-16")
SPEC["waist_pitch_joint"] = _flat(MUJOCO_PEAK["waist_pitch"], 37.0, "N5020-16")


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
