#!/usr/bin/env bash
set -euo pipefail
lab_source="vendor/unitree_rl_lab"
lab_revision="4960b84732b0c2ec593dccbfe963fda1bcd7b1e3"
mj_source="vendor/unitree_mujoco"
mj_revision="1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d"
if [ ! -d "$lab_source/.git" ]; then
  git clone --filter=blob:none --no-checkout https://github.com/unitreerobotics/unitree_rl_lab.git "$lab_source"
fi
git -C "$lab_source" fetch --depth 1 origin "$lab_revision"
git -C "$lab_source" checkout --detach "$lab_revision"
if [ ! -d "$mj_source/.git" ]; then
  git clone --filter=blob:none --sparse https://github.com/unitreerobotics/unitree_mujoco.git "$mj_source"
fi
git -C "$mj_source" sparse-checkout set unitree_robots/g1
git -C "$mj_source" fetch --depth 1 origin "$mj_revision"
git -C "$mj_source" checkout --detach "$mj_revision"
