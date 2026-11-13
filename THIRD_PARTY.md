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

## Optional Unitree RL Lab baseline and integration

The optional `scripts/fetch_lab.sh` fetches the following into ignored `vendor/`:

- `unitreerobotics/unitree_rl_lab` at `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3` (Apache-2.0):
  https://github.com/unitreerobotics/unitree_rl_lab
- `unitreerobotics/unitree_mujoco` at `1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d` (BSD-3-Clause):
  https://github.com/unitreerobotics/unitree_mujoco

The 29-joint ONNX velocity policy, deployment metadata, SDK joint ordering, observations and
PD controller configuration belong to Unitree. `g1lab.lab_policy` independently implements
that deployment contract in Python. No policy weights or upstream robot assets from these
repositories are redistributed here. This is distinct from the older 12-leg-joint RL Gym teacher.
Our command generator prepares experimental training overrides; it does not replace upstream
Isaac Lab, RSL-RL or claim a new RL algorithm.

## Private RTX handoff archive

`scripts/package_5090.py` builds a local, ignored transfer archive containing the pinned public
Unitree RL Lab and G1 MuJoCo sources/assets/ONNX used in this project. Their license texts are
included under `licenses/` in the archive. This does not add vendor weights or Git object stores
to this repository; downstream use remains subject to the respective upstream licenses.
