#!/usr/bin/env bash
# Usage: measure_run.sh <envs> <iterations> <output-dir-name> [extra prepare_lab_fast args...]
# Runs the project launcher on the project GPU (pinned by UUID in g1lab.gpu) and samples it every 0.5 s.
# Also samples GPU0 and fails if any process of this project ever appears there.
set -uo pipefail
envs=$1; iters=$2; name=$3; shift 3
root=/home/toann/TUAN__HUMAN_NOID_G1_RUNNER
cd "$root/g1-5090-handoff/g1-robust-locomotion-lab"
source "$root/envs/isaac51/bin/activate"
unset CUDA_VISIBLE_DEVICES  # g1lab.gpu.pin_cuda sets the UUID itself
export OMNI_KIT_ACCEPT_EULA=YES
gpu1=GPU-20e5a706-6231-50a3-4c9b-8e8d4db3c470
gpu0=GPU-4bb4ac0c-5f15-959c-dc6e-d6b9632cd45f
usd=$root/assets/unitree_model/G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd
mkdir -p artifacts/measure
samples=artifacts/measure/$name-gpu.csv
nvidia-smi -i $gpu1 --query-gpu=timestamp,memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits -lms 500 > "$samples" &
smi=$!
guard=artifacts/measure/$name-gpu0-guard.txt
( while true; do
    for pid in $(nvidia-smi -i $gpu0 -q -d PIDS | awk '/Process ID/ {print $NF}'); do  # compute + graphics
      cmd=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
      cwd=$(readlink /proc/$pid/cwd 2>/dev/null)
      case "$cmd $cwd" in *TUAN__HUMAN_NOID_G1_RUNNER*) echo "VIOLATION pid=$pid cwd=$cwd $cmd" >> "$guard";; esac
    done
    sleep 0.5
  done ) &
guard_pid=$!
python scripts/prepare_lab_fast.py --usd "$usd" --envs "$envs" --iterations "$iters" --output "artifacts/$name" --execute "$@"
rc=$?
kill $smi $guard_pid
if [ -s "$guard" ]; then echo "ERROR: project process seen on GPU0, see $guard" >&2; rc=97; else echo "GPU0 guard: no project process seen on GPU0"; fi
python - "$samples" "artifacts/$name" <<'PY'
import json, re, sys
rows = [l.split(", ") for l in open(sys.argv[1]) if l.count(",") == 3]
mem = [float(r[1]) for r in rows]; util = [float(r[2]) for r in rows]; power = [float(r[3]) for r in rows]
log = open(sys.argv[2] + "/train.log").read()
steps = [float(x) for x in re.findall(r"Computation: (\d+) steps/s", log)]
iters = [float(x) for x in re.findall(r"Iteration time: ([\d.]+)s", log)]
rewards = [float(x) for x in re.findall(r"Mean reward: (-?[\d.]+|nan)", log)]
lengths = [float(x) for x in re.findall(r"Mean episode length: ([\d.]+)", log)]
tail = steps[len(steps) // 2:] or steps
out = {
    "peak_gpu_mem_mib": max(mem), "mean_gpu_util_pct": sum(util) / len(util), "peak_power_w": max(power),
    "steps_per_s_median_2nd_half": sorted(tail)[len(tail) // 2] if tail else None,
    "iteration_time_s_median": sorted(iters)[len(iters) // 2] if iters else None,
    "iterations_logged": len(iters), "nan_in_log": bool(re.search(r"\bnan\b", log, re.IGNORECASE)),
    "first_reward": rewards[0] if rewards else None, "last_reward": rewards[-1] if rewards else None,
    "last_episode_length": lengths[-1] if lengths else None,
}
json.dump(out, open(sys.argv[2] + "/measure.json", "w"), indent=2)
print(json.dumps(out, indent=2))
PY
exit $rc
