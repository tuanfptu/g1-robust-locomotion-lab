import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from g1lab.config import EnvConfig
from g1lab.env import G1Env
from g1lab.training import check_bundle, load_policy, policy_action, save_bundle


@pytest.mark.parametrize("include_velocity", [False, True])
def test_policy_roundtrip_keeps_normalization(tmp_path, include_velocity):
    vec = VecNormalize(DummyVecEnv([lambda: G1Env(EnvConfig(linear_velocity_observation=include_velocity))]))
    try:
        model = PPO(
            "MlpPolicy", vec, n_steps=8, batch_size=8, n_epochs=1, policy_kwargs={"net_arch": [16]}, seed=1
        )
        model.learn(8)
        obs, _ = vec.envs[0].reset(seed=10)
        action = policy_action(model, vec, obs)
        save_bundle(model, vec, tmp_path / "bundle")
        loaded, normalized, raw = load_policy(tmp_path / "bundle", EnvConfig())
        assert raw.config.linear_velocity_observation == include_velocity
        try:
            np.testing.assert_allclose(action, policy_action(loaded, normalized, obs))
            assert not normalized.training
            assert not normalized.norm_reward
        finally:
            normalized.close()
        with pytest.raises(ValueError):
            check_bundle(tmp_path / "missing")
    finally:
        vec.close()
