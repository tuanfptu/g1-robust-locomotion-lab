"""Full-body recovery task. A fallen posture is a starting state, not termination."""

import mujoco
import numpy as np


def reset_recovery(env):
    # Settle a horizontally oriented robot using real contacts, not a guessed floor height.
    angle = {"supine": -np.pi / 2, "prone": np.pi / 2, "side": np.pi / 2}[env.config.recovery_pose]
    axis = np.array([1.0, 0.0, 0.0]) if env.config.recovery_pose == "side" else np.array([0.0, 1.0, 0.0])
    env.data.qpos[2] = 1.0
    env.data.qpos[3:7] = np.r_[np.cos(angle / 2), axis * np.sin(angle / 2)]
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)
    for _ in range(round(1.5 / env.config.physics_dt)):
        mujoco.mj_step(env.model, env.data)
    if not np.isfinite(env.data.qpos).all():
        raise FloatingPointError("Invalid recovery reset")
    env.data.qvel[:] = 0
    env.data.time = 0
    env.recovery_hold = 0.0
    env.spawn_x = float(env.data.qpos[0])
    mujoco.mj_forward(env.model, env.data)


def recovery_reward(env, action, info):
    linear, angular = env.body_velocity()
    upright = float(info["upright"])
    height = float(info["base_height_m"])
    stable = (
        upright > 0.9
        and 0.65 < height < 0.95
        and np.linalg.norm(linear) < 0.3
        and np.linalg.norm(angular) < 0.5
        and info["contacting_feet"] == 2
    )
    env.recovery_hold = env.recovery_hold + env.config.control_dt if stable else 0.0
    success = env.recovery_hold >= 1.0 - 1e-8
    terms = {
        "height": 2 * float(np.clip(height / 0.75, 0, 1)),
        "upright": 1.5 * upright,
        "standing": 2.0 if stable else 0.0,
        "posture": -0.1 * float(np.mean((env.data.qpos[env.qadr] - env.nominal) ** 2)),
        "power": -0.0002 * info["energy_j"] / env.config.control_dt,
        "speed": -0.02 * float(np.dot(angular, angular)),
        "success": 10.0 if success else 0.0,
    }
    # Locomotion diagnostics remain available, but are not recovery reward components.
    for key in list(info):
        if key.startswith("reward_"):
            del info[key]
    info.update({f"reward_{key}": float(value) for key, value in terms.items()})
    info.update(recovery_success=success, recovery_stable=bool(stable), recovery_hold_s=env.recovery_hold)
    return float(sum(terms.values())), success
