"""G1 29-DoF velocity task with the PD gains of the public v0 ONNX policy. Import only inside Isaac Sim."""

import importlib

from isaaclab.utils import configclass

_vendor = importlib.import_module("unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_env_cfg")


@configclass
class V0GainsEnvCfg(_vendor.RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        actuators = dict(self.scene.robot.actuators)
        # Replace (not mutate) so the vendor module-level robot config stays untouched.
        actuators["N5020-16"] = actuators["N5020-16"].replace(
            stiffness={
                ".*_shoulder_.*": 40.0,
                ".*_elbow_.*": 40.0,
                ".*_wrist_roll.*": 40.0,
                ".*_ankle_.*": 40.0,
                "waist_roll_joint": 200.0,
                "waist_pitch_joint": 200.0,
            },
            damping={
                ".*_shoulder_.*": 10.0,
                ".*_elbow_.*": 10.0,
                ".*_wrist_roll.*": 10.0,
                ".*_ankle_.*": 2.0,
                "waist_.*_joint": 5.0,
            },
        )
        actuators["W4010-25"] = actuators["W4010-25"].replace(damping=10.0)
        self.scene.robot = self.scene.robot.replace(actuators=actuators)
