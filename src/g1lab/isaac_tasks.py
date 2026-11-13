"""Isaac Lab task variants owned by this project.

Registration uses string entry points only, so importing this module needs gymnasium but not Isaac Sim;
the configuration module is imported lazily after the simulator app has started.
"""

import gymnasium as gym

VENDOR_TASK = "Unitree-G1-29dof-Velocity"
# Name must contain "Unitree" because the pinned train.py restricts --task choices to such ids.
V0_GAINS_TASK = "Unitree-G1-29dof-Velocity-G1LabV0Gains"
SPRINT_TASK = "Unitree-G1-29dof-Sprint-G1Lab"

# PD gains published with the public velocity/v0 ONNX policy (deploy.yaml, Unitree SDK joint order).
# The pinned vendor robot config differs: waist roll/pitch kp 40 instead of 200, arm/wrist kd 1 instead of 10.
V0_STIFFNESS_SDK = [100.0, 100.0, 100.0, 150.0, 40.0, 40.0] * 2 + [200.0, 200.0, 200.0] + [40.0] * 14
V0_DAMPING_SDK = [2.0, 2.0, 2.0, 4.0, 2.0, 2.0] * 2 + [5.0, 5.0, 5.0] + [10.0] * 14


def register():
    if V0_GAINS_TASK in gym.registry:
        return
    gym.register(
        id=SPRINT_TASK,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "g1lab.isaac_sprint:SprintEnvCfg",
            "play_env_cfg_entry_point": "g1lab.isaac_sprint:SprintEnvCfg",
            "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
        },
    )
    gym.register(
        id=V0_GAINS_TASK,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "g1lab.isaac_v0_cfg:V0GainsEnvCfg",
            "play_env_cfg_entry_point": "g1lab.isaac_v0_cfg:V0GainsEnvCfg",
            "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
        },
    )
