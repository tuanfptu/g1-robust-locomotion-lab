import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from g1lab.config import EnvConfig
from g1lab.env import G1Env


@pytest.mark.parametrize("terrain", ["flat", "slope", "rough", "stairs"])
def test_seeded_rollout_and_terrain(terrain):
    env = G1Env(EnvConfig(terrain=terrain, difficulty=0.3))
    try:
        first, _ = env.reset(seed=7)
        second, _ = env.reset(seed=7)
        np.testing.assert_array_equal(first, second)
        assert env.model.nu == 29
        assert env.observation_space.contains(first)
        assert abs(env.ground_height(0, 0)) < 0.01
        for _ in range(20):
            obs, reward, done, truncated, info = env.step(np.zeros(29))
            assert np.isfinite(obs).all() and np.isfinite(reward)
            assert info["energy_j"] >= 0
            assert info["upright"] == pytest.approx(env.data.xmat[env.pelvis].reshape(3, 3)[2, 2])
            assert info["reward_upright"] == pytest.approx(0.5 * info["upright"])
            if done or truncated:
                env.reset(seed=7)
        if terrain == "stairs":
            assert env.ground_height(2, 0) > 0.05
        if terrain == "slope":
            assert env.ground_height(1, 0) > 0.01
    finally:
        env.close()


def test_gym_api():
    env = G1Env()
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()


def test_impulse_integrates_and_time_limit():
    env = G1Env(
        EnvConfig(random_commands=False, push_at=0, push_duration=0.04, push_impulse=2, episode_seconds=0.04)
    )
    try:
        env.reset(seed=5)
        _, _, _, truncated, first = env.step(np.zeros(29))
        assert not truncated
        _, _, _, truncated, second = env.step(np.zeros(29))
        assert truncated
        assert first["impulse_ns"] + second["impulse_ns"] == pytest.approx(2)
    finally:
        env.close()


def test_randomization_does_not_accumulate():
    env = G1Env(EnvConfig(domain_randomization=True))
    try:
        env.reset(seed=4)
        masses = env.model.body_mass.copy()
        env.reset(seed=4)
        np.testing.assert_array_equal(masses, env.model.body_mass)
        assert not np.array_equal(masses, env.base_mass)
    finally:
        env.close()


def test_invalid_configuration_and_action():
    with pytest.raises(ValueError):
        EnvConfig(difficulty=2)
    with pytest.raises(ValueError):
        EnvConfig(control_dt=0.003)
    env = G1Env()
    try:
        env.reset(seed=1)
        with pytest.raises(ValueError):
            env.step(np.full(29, np.nan))
    finally:
        env.close()
