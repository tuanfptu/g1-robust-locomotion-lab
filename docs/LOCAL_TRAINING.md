# Local mixed-terrain PPO pilot

This continues the Unitree-teacher → DAgger → PPO+BC work. It is not training walking from scratch. Native MuJoCo physics and the small SB3 network run on CPU; the 1660 Ti renders the viewer. A larger GPU alone will not accelerate this physics backend.

## Reproduce

Fetch the optional teacher using `scripts/fetch_teacher.sh`, then reproduce the bootstrap/DAgger checkpoints as described in EXPERIMENT_1660.md, or use the paired local checkpoints. From the repository root:

```bash
MPLCONFIGDIR=/tmp/g1lab-mpl .venv/bin/python scripts/mixed_campaign.py \
  --bundle runs/autonomous-1660/selected-terrain \
  --run runs/mixed-1660-v1 \
  --data runs/autonomous-1660/dagger-42/aggregated-data.npz \
  --data runs/autonomous-1660/dagger-slope/aggregated-data.npz
```

Use a new run directory on each invocation. This local command needs the prior checkpoint and teacher-labelled datasets, which are not stored in Git. Three worker environments contribute flat, 2° slope, and approximately 1.2 cm roughness samples to each PPO rollout, with an equal number of steps per worker. Training command is fixed at 0.3 m/s forward for this first pilot. These task parameters are intentionally limited; this does not establish arbitrary command following on terrain.

Default budget: three rounds of at least 16,384 transitions, rounded up by SB3 to complete a rollout. PPO learning rate 5e-6; frozen normalization; initial 1,536 steps of critic warmup each round; teacher replay of up to 16,000 samples from each source. The replay includes prior flat and slope DAgger states, not a new rough-terrain expert dataset.

Validation uses seeds 6000–6001 and 15-second episodes on all three terrains. A candidate must preserve survival rate on every terrain, add no more than 0.03 m/s tracking RMSE on any terrain, and improve the equal-task score mean(survival − RMSE). This is a practical selection heuristic, not a statistical significance test. Rejected candidates do not replace the last accepted model. The final selected model is evaluated once using seeds 40000–40002, 30-second episodes. Those seeds do not influence selection. All candidates, including rejected runs, remain on disk.

`progress.json`, `selection.json`, `heldout.json`, per-episode CSVs, normalization statistics and policy weights provide an audit trail. Checkpoints also record the complete worker environment mix in `environment_mix.json`; `env_config.json` is the first worker's policy contract/default task, not a description of all training terrains.

## When to move to the larger machine

Continue environment development, short pilots, regression checks, and viewer work locally. Before a longer experiment, use the measured `training.json` throughput to estimate total wall time: transitions / steps_per_second × number of runs. Move the campaign when its projected time exceeds the available local overnight window, or measured memory use makes the local machine impractical.

To benefit from the RTX 5090 for simulation throughput, first implement and validate a GPU physics backend, including contact/dynamics and policy-input parity checks against this CPU reference. Benchmark that backend on the destination machine before promising speedups. CUDA PyTorch alone does not convert native MuJoCo into GPU physics. Multi-seed ablations and larger terrain sweeps are the first workloads to move; preserve the CPU reference tests.

## Skill-preservation follow-up

The first terrain-only pilot improved forward tracking but lost in-place turning in the broader audit. It is kept as a candidate and is not the default demo. A follow-up addresses the missing training/selection coverage:

```bash
MPLCONFIGDIR=/tmp/g1lab-mpl .venv/bin/python scripts/preservation_pilot.py \
  --bundle runs/autonomous-1660/selected-terrain \
  --run runs/preservation-1660-v2 --validation-episodes 1 --validation-seconds 15
```

This collects 1,536 state/action pairs per task from the previous competent student, for 9,216 behavioral replay examples across flat forward, slope, rough, stand, lateral, and turn. These labels come from the existing student, not new Unitree expert rollouts. Six workers train one shared policy on all six tasks. Each round has 18,432 PPO transitions, with the first 6,144 warming the critic. Selection checks all six tasks at seed 7000 for 15 s, preserving survival, bounding velocity RMSE increases by 0.03 m/s and yaw RMSE increases by 0.05 rad/s, and requiring improved equal-task score. This is still a short validation heuristic; final 30-second tests use fresh seeds 50000–50002.

A single validation seed and one training trajectory do not establish algorithm superiority. The pilot tests whether preserving prior-policy behaviors avoids the specific turning regression; a controlled teacher-replay versus student-replay ablation with independent training seeds remains future work.

The completed follow-up still regressed on 30-second turning tests (2/3 versus 3/3); it is not promoted. Future invocations now default to two 30-second validation episodes per task. The command above retains the original shorter settings solely for reproducing the completed negative result. A longer gate reduces this specific blind spot but does not guarantee generalization.
