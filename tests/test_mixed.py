import json

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from g1lab.config import EnvConfig
from g1lab.selection import acceptable
from g1lab.training import make_env, save_bundle


def test_mixed_rollout_and_bundle_provenance(tmp_path):
    configs = [EnvConfig(terrain=t, difficulty=0.2) for t in ("flat", "slope", "rough")]
    vec = VecNormalize(DummyVecEnv([make_env(c, 42, i) for i, c in enumerate(configs)]))
    try:
        model = PPO(
            "MlpPolicy", vec, n_steps=4, batch_size=12, n_epochs=1, policy_kwargs={"net_arch": [16]}, seed=42
        )
        model.learn(12)
        save_bundle(model, vec, tmp_path / "bundle")
        mix = json.loads((tmp_path / "bundle/environment_mix.json").read_text())
        assert [item["terrain"] for item in mix] == ["flat", "slope", "rough"]
        assert model.num_timesteps == 12
    finally:
        vec.close()


def test_selection_rejects_terrain_forgetting():
    def metrics(survival, error):
        return {"survival_rate": survival, "velocity_rmse_mps": {"mean": error}}

    baseline = {t: metrics(1, 0.15) for t in ("flat", "slope", "rough")}
    candidate = {t: metrics(1, 0.10) for t in baseline}
    assert acceptable(candidate, baseline)
    candidate["rough"] = metrics(0.5, 0.05)
    assert not acceptable(candidate, baseline)
    candidate["rough"] = metrics(1, 0.19)
    assert not acceptable(candidate, baseline)


def test_selection_rejects_yaw_regression_when_requested():
    baseline = {
        "turn": {"survival_rate": 1, "velocity_rmse_mps": {"mean": 0.1}, "yaw_rmse_radps": {"mean": 0.15}}
    }
    candidate = {
        "turn": {"survival_rate": 1, "velocity_rmse_mps": {"mean": 0.09}, "yaw_rmse_radps": {"mean": 0.25}}
    }
    assert acceptable(candidate, baseline)
    assert not acceptable(candidate, baseline, yaw_tolerance=0.05)
