#!/usr/bin/env bash
set -euo pipefail
task_source="vendor/unitree_rl_gym"
task_revision="276801e46c5d433564f24658bac64f254b7d2d4b"
if [ ! -d "$task_source/.git" ]; then
  git clone --filter=blob:none --no-checkout https://github.com/unitreerobotics/unitree_rl_gym.git "$task_source"
fi
git -C "$task_source" sparse-checkout init --cone
git -C "$task_source" sparse-checkout set deploy/deploy_mujoco deploy/pre_train/g1 resources/robots/g1_description
git -C "$task_source" fetch --depth 1 origin "$task_revision"
git -C "$task_source" checkout --detach "$task_revision"
