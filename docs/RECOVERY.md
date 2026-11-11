# Recovery milestone: supine stand-up

The walking policy is retained separately. Recovery is a new full-body task with a randomly initialized PPO policy; no Unitree teacher or walking checkpoint is used for this pilot. This does not yet implement running, controlled falling, an automatic locomotion/recovery switch, or a converged stand-up controller.

## Environment

Use `configs/recovery-supine.json`. All 29 actuators are controlled with residual position targets and native model PD gains/limits. Each episode starts with the body rotated into the requested lying orientation and dropped onto the flat ground. The simulator settles contacts for 1.5 simulated seconds under nominal PD before the episode begins. Episode time is then zeroed and joint/root velocities are cleared. Seeded joint perturbations are preserved. Reset settling costs wall time but is excluded from episode reward/time.

Supported starting orientations: supine, prone, side. Only supine is trained in this first pilot. These resets are controlled test starts, not a distribution of realistic dynamic falls.

A lying body does not terminate the episode. Time limit: 12 seconds. Success requires **all** of the following for one uninterrupted second:

- Pelvis vertical-axis cosine above 0.9.
- Pelvis height between 0.65 and 0.95 m above the floor.
- Pelvis linear speed below 0.3 m/s and angular speed below 0.5 rad/s.
- Both feet in ground contact.

Success terminates the episode. The criterion does not yet require hands to leave the floor; additional support/contact validation is needed before certifying unsupported standing. A non-finite simulation state raises an error rather than being counted as recovery.

Reward per 0.02-second control step:

- Height: `2 * clip(pelvis_height / 0.75, 0, 1)`.
- Upright: `1.5 * upright_cosine`.
- Stable-standing conditions: +2 while all conditions hold.
- Posture: `-0.1 * mean((joint_position - nominal)^2)`.
- Mechanical power proxy: `-0.0002 * energy_per_step / control_dt`.
- Angular speed: `-0.02 * angular_speed_squared`.
- Successful hold: +10 once, followed by termination.

This is initial reward shaping, not a validated solution. It may encourage partial rising without complete recovery. It does not directly penalize head impacts or jerk yet, and the power estimate is not electrical power. Video and contact inspection remain necessary alongside the success metric.

## Reproduce the first pilot

```bash
MPLCONFIGDIR=/tmp/g1lab-mpl .venv/bin/g1lab train \
  --config configs/recovery-supine.json --run runs/recovery-supine-1660-v1 \
  --steps 65536 --envs 2 --workers --threads 1 --checkpoint-every 16384 \
  --learning-rate 0.0001 --log-std-init -0.5

MPLCONFIGDIR=/tmp/g1lab-mpl .venv/bin/python scripts/evaluate_recovery.py \
  --bundle runs/recovery-supine-1660-v1/final \
  --output artifacts/recovery-supine-1660-v1/final-evaluation --episodes 5 --seed 60000
```

Use a fresh output/run directory for each invocation. `g1lab eval` intentionally rejects recovery checkpoints because locomotion survival metrics are not stand-up success metrics. Use the dedicated script. Omitting `--bundle` evaluates nominal PD. Add `--video` with a working MuJoCo renderer to save the first rollout.

```bash
MUJOCO_GL=egl EGL_PLATFORM=surfaceless MPLCONFIGDIR=/tmp/g1lab-mpl \
  .venv/bin/python scripts/evaluate_recovery.py \
  --bundle runs/recovery-supine-1660-v1/final \
  --output artifacts/recovery-supine-1660-v1/video --episodes 1 --seed 62000 --video
```

The final checkpoint is selected by the fixed step budget, not by test-seed performance. Seeds 61000–61002 are development probes; 60000–60004 compare final policy with PD. One training seed (42) is a pipeline pilot, not a reliability study. Report actual success even when zero. A suitable next experiment is a curriculum of easier near-standing states before full supine recovery, while keeping full-supine evaluation unchanged.

## Completed local pilot result

65,536 transitions, 241.54 seconds of training, 271.33 transitions/s on the existing CPU backend. Final budget-selected policy: **0/5 successful recoveries** on seeds 60000–60004. Nominal PD: **0/5** on the same seeds. Mean maximum pelvis height was 0.105 m for the learned policy versus 0.061 m for PD; the learned policy still remained near the floor. Increased shaped reward is not evidence of learning to stand.

Development probes at 16,384 and 32,768 transitions were also 0/3. Full-body supine recovery has **not** been solved by this pilot. The locomotion default is unchanged. Compact protocol, training and per-episode evidence are in `experiment-data/recovery-supine-1660-v1/`; model weights remain local under `runs/recovery-supine-1660-v1/final`.

## Combined stress-test video

`scripts/sprint_recovery_video.py` uses the current locomotion and recovery checkpoints to render a real-physics rollout, without retraining or animation. Forward command starts at zero, then 0.3 m/s, then ramps toward 2 m/s. A 40 Ns lateral stress impulse is scheduled at 14 s only if the locomotion controller has not already fallen. Physical qpos/qvel and simulation time are copied across the controller handoff; the robot is not reset to a convenient recovery pose. The target controller has its own PD gains and observation/action normalization. This transition is a prototype, not a validated hybrid controller.

At seed 63000, the robot fell at 11.00 s, before the scheduled impulse. Peak horizontal speed measured before the fall was 1.843 m/s; this is a transient peak, not sustained running performance or verified running gait. The recovery pilot did not stand up during the remainder of the 26-second video. This combined attempt starts from the actual fall, unlike the supine-only recovery training distribution.

```bash
MUJOCO_GL=egl EGL_PLATFORM=surfaceless MPLCONFIGDIR=/tmp/g1lab-mpl \
  .venv/bin/python scripts/sprint_recovery_video.py \
  --output artifacts/sprint-recovery-pilot-v1
```

The MP4 and trajectory are local under that output directory. Compact summary/protocol are retained in `experiment-data/sprint-recovery-pilot-v1/`. This is a demonstration of current limitations, not a successful sprint-and-get-up demonstration.
