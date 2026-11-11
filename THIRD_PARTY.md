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
