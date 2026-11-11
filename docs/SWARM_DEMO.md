# Local parallel-world G1 demonstration

This is an **inference-only** demonstration of 16 independent MuJoCo worlds, displayed together in one overview. It does not train a policy, use Isaac Sim, or use MuJoCo Warp. Each world has separate mutable `MjData` and a separate copy of the original Unitree recurrent policy, including its own LSTM state. A read-only model is shared across worlds; four worker threads advance worlds concurrently on CPU. The NVIDIA GPU renders the composite view.

The policy is the pinned Unitree `motion.pt`, on its original 12-controlled-leg-DoF G1 model with a fixed upper body. Robots in the overview do not collide with each other: their states come from different worlds. This is not a coordinated swarm or multi-agent learning task.

## What the video shows

- Blue: forward target 0.5 m/s.
- Green: target 1.0 m/s.
- Yellow: target 1.6 m/s.
- Red: target 2.5 m/s, intentionally beyond the tested tracking envelope.

Commands begin after one simulated second and ramp over two seconds. Each world uses initial joint jitter from its own seeded random generator (88000 + world index). A fallen world continues for about half a simulated second to show the fall, then resets independently. Episode horizons are staggered between 8 and 9.2 seconds. A reset is a reset, **not** a learned stand-up behavior. No policy weights change.

Physics step: 0.002 s. Control step: 0.02 s. Video: 1280×720 at 25 fps, displaying simulation time at normal speed. Offline simulation/rendering may take longer than the final playback duration.

## Reproduce

Fetch the optional pinned vendor assets with `scripts/fetch_teacher.sh` and use a fresh output directory:

```bash
MUJOCO_GL=egl EGL_PLATFORM=surfaceless MPLCONFIGDIR=/tmp/g1lab-mpl \
  .venv/bin/python scripts/swarm_demo.py \
  --output artifacts/swarm-1660-v2 --worlds 16 --workers 4 --seconds 12
```

Outputs: `swarm.mp4`, `protocol.json`, `summary.json`, and `timeline.csv`. The overview adds the dynamic rendering geometry from each independent state to a single render scene; robot motion itself is computed separately, not duplicated from a recorded trajectory.

A local integration check confirmed that stepping one world leaves another world's physical state unchanged, and a 50-step threaded rollout matched its serial counterpart to tolerance 1e-10 in joint/root positions and velocities. This does not establish performance or correctness for every world count and driver combination.

To scale to hundreds or thousands of environments for actual RL, the next implementation needs a GPU physics backend and batched policy/PPO pipeline. Merely showing more robot meshes is not that migration.

## Measured local result

16 worlds, 4 CPU worker threads, 12 simulated seconds per world, 300 video frames. Collection and rendering took 24.72 s excluding initialization. Aggregate throughput was 790 control transitions/s excluding rendering and 388/s including rendering. The latter corresponds to 0.49× real-time per world. Four falls and 16 individual resets occurred. Learning updates: **zero**. These timings are one local run, not a scaling benchmark.

The video was visually inspected to confirm that all 16 robot states fit in the overview. Compact data is in `experiment-data/swarm-1660-v2/`; the MP4 remains local at `artifacts/swarm-1660-v2/swarm.mp4`.
