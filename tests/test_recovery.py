import numpy as np
import pytest

from g1lab.config import EnvConfig
from g1lab.env import G1Env


@pytest.mark.parametrize("pose", ["supine", "prone", "side"])
def test_recovery_reset_is_seeded_and_falling_does_not_end_episode(pose):
    env = G1Env(
        EnvConfig(
            task="recovery", recovery_pose=pose, random_commands=False, command=(0, 0, 0), episode_seconds=0.1
        )
    )
    try:
        a, _ = env.reset(seed=42)
        b, _ = env.reset(seed=42)
        np.testing.assert_allclose(a, b)
        assert env.data.time == 0
        for step in range(5):
            obs, reward, terminated, truncated, info = env.step(np.zeros(29))
            assert np.isfinite(obs).all() and np.isfinite(reward)
            assert not terminated
            assert not info["recovery_success"]
            assert "reward_fall" not in info
            assert truncated == (step == 4)
        assert info["fallen"]
    finally:
        env.close()


def test_recovery_requires_full_body_and_flat_ground():
    with pytest.raises(ValueError):
        EnvConfig(task="recovery", leg_only=True)
    with pytest.raises(ValueError):
        EnvConfig(task="recovery", terrain="stairs")


def test_success_requires_uninterrupted_stable_hold():
    from g1lab.recovery import recovery_reward

    env = G1Env(EnvConfig(task="recovery"))
    try:
        env.reset(seed=42)
        # Isolate the hold logic with controlled measured state/diagnostics.
        env.body_velocity = lambda: (np.zeros(3), np.zeros(3))
        info = {"upright": 1.0, "base_height_m": 0.75, "contacting_feet": 2, "energy_j": 0.0}
        for _ in range(49):
            _, success = recovery_reward(env, np.zeros(29), info)
            assert not success
        info["contacting_feet"] = 1
        assert not recovery_reward(env, np.zeros(29), info)[1]
        assert env.recovery_hold == 0
        info["contacting_feet"] = 2
        for _ in range(50):
            _, success = recovery_reward(env, np.zeros(29), info)
        assert success
    finally:
        env.close()


def test_recovery_checkpoint_restores_task_and_rejects_locomotion_evaluator(tmp_path):
    from types import SimpleNamespace

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from g1lab.evaluation import evaluate
    from g1lab.training import load_policy, save_bundle

    vec = VecNormalize(DummyVecEnv([lambda: G1Env(EnvConfig(task="recovery", action_scale=1.5))]))
    try:
        model = PPO("MlpPolicy", vec, n_steps=4, batch_size=4, policy_kwargs={"net_arch": [16]})
        save_bundle(model, vec, tmp_path / "checkpoint")
        _, restored, raw = load_policy(tmp_path / "checkpoint", EnvConfig())
        try:
            assert raw.config.task == "recovery"
            assert raw.config.action_scale == 1.5
            raw.reset(seed=42)
            assert not raw.step(np.zeros(29))[2]
        finally:
            restored.close()
        with pytest.raises(ValueError, match="evaluate_recovery"):
            evaluate(SimpleNamespace(bundle=str(tmp_path / "checkpoint")))
    finally:
        vec.close()
