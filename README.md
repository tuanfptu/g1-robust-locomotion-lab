# G1 Sprint — learning to run fast on Unitree G1 (simulation)

A PPO policy that makes the 29-DoF Unitree G1 **sprint with a real flight phase, hold speed, slow down and
stop** — trained in Isaac Lab, verified in an independent simulator (MuJoCo), within the motor torque and
speed limits.

![Walk at 1 m/s, then sprint to 5.5 m/s: Unitree baseline (left) falls, ours (right) accelerates, holds and stops](docs/media/demo-cruise-to-5.5mps.gif)

Demo videos (MuJoCo, same seed; walk 1 m/s for 5 s → accelerate → hold 8 s → decelerate → stop):
**[1 → 3.5 m/s](docs/media/demo-cruise-to-3.5mps.mp4)** · **[1 → 5.5 m/s](docs/media/demo-cruise-to-5.5mps.mp4)**.
The baseline falls at t = 9.4 s in both, as the command passes ~2.8 m/s; our policy holds 3.44 / 4.94 m/s
with 41–42 % flight time, then stops.

## Results

MuJoCo, 10 unseen seeds, stand → accelerate 2 m/s² → **hold 60 s** → decelerate → stop, same actuator model for
both policies.

| | Unitree public ONNX (baseline) | **Ours** |
|---|---|---|
| Fastest command passed by 10/10 seeds | 1.0 m/s | **5.5 m/s (4.94 m/s actual)** |
| At 2.0 m/s and above | falls (10/10) within ~6 s | runs, holds and stops |
| Gait | walking, 0 % flight | running, **46 % flight**, 0.13 s ground contact |
| Peak joint speed / rated | — | 72 % |

Pass = no fall, vx ≥ 85 % of command, yaw rate RMS < 0.25 rad/s, heading error < 30°, final speed < 0.15 m/s,
no joint above rated speed. Raw data: [docs/experiment-data/sprint-v2/](docs/experiment-data/sprint-v2/).

Not yet achieved: forefoot strike (lands on heels), active arm swing, upright posture, cadence rising with speed.

## What changed vs. the Unitree baseline

- **Task** `Unitree-G1-29dof-Sprint-G1Lab` ([isaac_sprint.py](src/g1lab/isaac_sprint.py)): same observations and
  actions (480 → 29); acceleration-limited speed commands (start, accelerate, hold, brake, stop); speed
  curriculum up to 11 m/s.
- **Reward**: walking terms fade out above 1.6–2.2 m/s; 20 running terms added — flight phase, short ground
  contact, speed-dependent cadence, horizontal push-off (Σ mᵢaᵢ), foot placement, knee lift, ankle
  dorsiflexion, low whole-body angular momentum, acceleration-dependent lean, heading, actuator envelope, fall
  penalty. Running bonuses are paid only in proportion to speed tracking, so jogging slowly cannot farm them.
- **Actuators** ([actuator_spec.py](src/g1lab/actuator_spec.py)), shared by Isaac and MuJoCo: MuJoCo-model peak
  torques plus Unitree N7520 torque–speed derating, 0–20 ms latency.
- **Domain randomization**: friction, link mass, torso mass/CoM, PD gains, armature, random pushes.
- **Autopilot** ([autopilot_sprint.py](scripts/autopilot_sprint.py)): unattended train → evaluate → keep best →
  roll back on collapse loop, offline, with an HTML dashboard.

Full write-up (Vietnamese): [docs/G1_SPRINT_REPORT.md](docs/G1_SPRINT_REPORT.md).

## Reproduce

**1. Evaluate the released policy (CPU, ~10 min).** Python 3.10:

```bash
python3.10 -m venv .venv-eval
.venv-eval/bin/pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.venv-eval/bin/pip install -e '.[dev,lab]' -c requirements-lock.txt
bash scripts/fetch_lab.sh            # pinned unitree_rl_lab + unitree_mujoco into vendor/

# 10 seeds x 60 s speed ladder for ours (models/) and the baseline
.venv-eval/bin/python scripts/mujoco_sprint_eval.py --policy-dir models/g1-sprint-v2-r11 \
  --speeds 1,2,3,4,5,5.5 --seeds 10 --seed0 99000 --hold 60 --output ours.json
.venv-eval/bin/python scripts/mujoco_sprint_eval.py \
  --policy-dir vendor/unitree_rl_lab/deploy/robots/g1_29dof/config/policy/velocity/v0 \
  --speeds 1,1.5,2,3 --seeds 10 --seed0 99000 --hold 60 --output baseline.json

# Side-by-side demo video (needs EGL)
.venv-eval/bin/python scripts/render_compare.py --right-policy models/g1-sprint-v2-r11 \
  --cruise 1.0 --speed 5.5 --hold 8 --output demo.mp4
```

**2. Train (NVIDIA GPU).** Python 3.11 with Isaac Sim 5.1.0, Isaac Lab v2.3.2, PyTorch 2.7.0+cu128, RSL-RL 3.1.2,
installed with [these constraints](docs/experiment-data/5090-bringup/isaac51-constraints.txt)
(steps: [docs/RTX_5090_BRINGUP.md](docs/RTX_5090_BRINGUP.md)). G1 USD from
[unitreerobotics/unitree_model](https://huggingface.co/datasets/unitreerobotics/unitree_model).

```bash
USD=/path/to/G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd

# From scratch, unattended (~6 h on one RTX 5090 to reach the released policy)
.venv-eval/bin/python scripts/autopilot_sprint.py --output artifacts/run1 --init-checkpoint scratch \
  --usd "$USD" --hours 0 --envs 4096 --round-iterations 600 --lr 1e-3 --start-level 1.0 --speed-limit 11

python3 scripts/autopilot_dashboard.py --run artifacts/run1 --watch 60   # open artifacts/run1/dashboard.html
echo '{"stop": true}' > artifacts/run1/control.json                       # stop after the current round
```

GPU pinning is in [src/g1lab/gpu.py](src/g1lab/gpu.py) (set `G1LAB_GPU_UUID` / `G1LAB_GPU_PCI` for your machine).
Autopilot rules: [docs/AUTOPILOT.md](docs/AUTOPILOT.md).

## Layout

```text
models/g1-sprint-v2-r11/   released policy: ONNX + deploy.yaml + RSL-RL checkpoint
src/g1lab/                 sprint task, actuator model, GPU pinning, MuJoCo runner and evaluation
scripts/                   autopilot, dashboard, Isaac/MuJoCo evaluation, export, demo rendering
docs/                      report, media, experiment data, older GTX 1660 / SB3 work
vendor/                    pinned unitree_rl_lab and unitree_mujoco (fetched, not committed)
```

Simulation only — no real-robot code. Third-party licenses: [THIRD_PARTY.md](THIRD_PARTY.md).
