# Third-party assets

`src/g1lab/assets/g1/` contains the unmodified `g1.xml`, meshes and BSD-3-Clause LICENSE
from Google DeepMind MuJoCo Menagerie, `unitree_g1/`:
https://github.com/google-deepmind/mujoco_menagerie

Source commit: `370c43965323c0944b433b202aac5058b5f907a9`.
The model derives from Unitree's public G1 description. Its license is retained alongside the files.
The lab adds terrain, changes mesh paths and geom groups in memory, and sets simulation timestep.
Position actuator gains and force limits are from the upstream basic G1 model; they are a simulation
starting point and do not certify physical robot fidelity.

Stable-Baselines3 supplies PPO. This project does not claim to introduce PPO or the G1 model.

## Optional Unitree teacher

The optional experiment scripts fetch `unitreerobotics/unitree_rl_gym` at commit
`276801e46c5d433564f24658bac64f254b7d2d4b` into ignored `vendor/`:
https://github.com/unitreerobotics/unitree_rl_gym

Its `deploy/pre_train/g1/motion.pt` is an upstream pretrained recurrent policy. Our adapter uses
the observation scaling, 0.8 s phase, PD gains and default leg pose from the accompanying
`deploy/deploy_mujoco/configs/g1.yaml` and deployment implementation.
The upstream policy and model files are not redistributed in this repository.
Student checkpoints in the local experiments are initialized through imitation of this teacher;
their success is not evidence of learning locomotion from scratch.
