# Validation on the development machine

Tested 2026-11-11, Linux, Python 3.10.12, Ryzen 5 4600H, 16 GB RAM,
GTX 1660 Ti Mobile 6144 MiB, NVIDIA driver 595.99.02.

## Passed

- `ruff check src tests`.
- 11 pytest tests: four terrain seeded rollouts, Gymnasium API, impulse integration/time limit,
  randomization reset, bad inputs, policy/normalization roundtrip, eval/report exports,
  rejection of mismatched evaluation protocols.
- PPO train: 1,024 steps, 128-step rollouts, one CPU environment, periodic paired checkpoints.
- Fine-tune resume: loaded 1,024-step checkpoint and trained to 1,280 steps with randomized physics.
- Multiprocess smoke: two SubprocVecEnv workers, 128 total steps.
- Full evaluation suite: 22 scenarios × 2 episodes, 2-second horizon, raw reward and trajectory exports.
- Six-stage curriculum smoke: 512 steps per stage (PPO rounds the requested 8 steps to a rollout),
  including transfer between flat, randomized, slope and rough environments; 3,072 total steps.
- Report: summary JSON/episode CSV rendered to PNG and Markdown.
- Video: all five flat scenarios, one episode each, 1-second horizon, MP4 output.
- EGL rendering outside sandbox reports `NVIDIA GeForce GTX 1660 Ti/PCIe/SSE2`, 640×480 RGB.
- Built a distributable wheel with build isolation; verified G1 XML/license/meshes and reset the
  environment from an extracted wheel, independently of the editable source package.
- Venv-only dependency check passes with system ROS `PYTHONPATH` removed.

Two expected Gymnasium warnings: unbounded observation Box. Observation values are checked for
finiteness and policy inputs use VecNormalize. No failure was suppressed.
ROS pytest autoload is disabled during tests to prevent unrelated system plugins from entering venv.

## Timing observations

- 1,000 zero-action CPU physics control steps: 2.67 s (~374 control steps/s), including resets.
- Initial PPO smoke: ~240 control steps/s with updates/checkpoint overhead, two torch CPU threads.
- Each control step contains 10 physics substeps. These numbers are neither GPU physics throughput
  nor a guarantee for longer experiments, terrain changes, thermal throttling, or other machines.

## Quality limits

The smoke policy is **not a trained walking controller**. Evaluation horizons of 1–2 s verify the
pipeline only. Survival over these short horizons must not be compared to the 30-second target.
No convergence, successful staircase traversal, robust push rejection, AMP, native GPU physics,
cross-training-seed confidence interval, or physical robot result is claimed.
Interactive viewer code is provided; automated checks cover offscreen rendering, not keyboard UI.

Raw smoke outputs remain in local ignored `runs/` and `artifacts/`. They are intentionally not
published as benchmark results. Repeat using `bash scripts/smoke.sh runs/a-new-name`.
