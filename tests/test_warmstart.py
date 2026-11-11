import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from g1lab.config import EnvConfig
from g1lab.env import G1Env
from g1lab.training import BehaviorAnchor, CriticWarmup, load_policy, save_bundle


def test_profile_roundtrip_and_upper_body_hold(tmp_path):
    config = EnvConfig(
        crouched_home=True,
        unitree_pd=True,
        phase_observation=True,
        linear_velocity_observation=True,
        leg_only=True,
        action_scale=1.0,
        control_dt=0.01,
    )
    vec = VecNormalize(DummyVecEnv([lambda: G1Env(config)]))
    raw = vec.envs[0]
    try:
        obs, _ = raw.reset(seed=42)
        assert obs.shape == (116,)
        raw.step(np.ones(29) * 0.05)
        np.testing.assert_array_equal(raw.data.ctrl[12:], raw.nominal[12:])
        model = PPO("MlpPolicy", vec, n_steps=8, batch_size=8, policy_kwargs={"net_arch": [16]})
        save_bundle(model, vec, tmp_path / "bundle")
        _, restored, env = load_policy(tmp_path / "bundle", EnvConfig())
        try:
            assert env.config.action_scale == 1
            assert env.config.leg_only
            assert env.config.unitree_pd
            assert env.config.control_dt == 0.01
            assert env.observation_space.shape == (116,)
        finally:
            restored.close()
    finally:
        vec.close()


def test_critic_warmup_and_anchor_can_train_together(tmp_path):
    vec = VecNormalize(DummyVecEnv([G1Env]))
    try:
        obs = vec.reset()
        np.savez(tmp_path / "data.npz", observations=np.repeat(obs, 8, axis=0), actions=np.zeros((8, 29)))
        model = PPO(
            "MlpPolicy", vec, n_steps=8, batch_size=8, n_epochs=1, policy_kwargs={"net_arch": [16]}, seed=42
        )
        model.learn(16, callback=[CriticWarmup(8), BehaviorAnchor(tmp_path / "data.npz")])
        assert all(p.requires_grad for p in model.policy.action_net.parameters())
    finally:
        vec.close()
