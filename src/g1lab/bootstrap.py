"""Initialize an SB3 policy from the officially shipped Unitree recurrent teacher.

This is imitation initialization, not from-scratch PPO. Teacher assets remain in vendor/.
"""

import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .config import EnvConfig
from .env import G1Env
from .training import save_bundle


def teacher_config(config):
    from dataclasses import replace

    return replace(
        config,
        action_scale=1.0,
        crouched_home=True,
        unitree_pd=True,
        leg_only=True,
        phase_observation=True,
        linear_velocity_observation=True,
    )


class UnitreeTeacher:
    def __init__(self, source):
        self.policy = torch.jit.load(str(Path(source) / "deploy/pre_train/g1/motion.pt"), map_location="cpu")
        self.policy.eval()
        self.reset()

    def reset(self):
        self.policy.reset_memory()
        self.previous = np.zeros(12)

    def action(self, env):
        rotation = env.data.xmat[env.pelvis].reshape(3, 3)
        phase = 2 * np.pi * (env.data.time % 0.8) / 0.8
        obs = np.concatenate(
            [
                env.data.qvel[3:6] * 0.25,
                rotation.T @ [0, 0, -1],
                env.command * [2, 2, 0.25],
                env.data.qpos[env.qadr[:12]] - env.nominal[:12],
                env.data.qvel[env.vadr[:12]] * 0.05,
                self.previous,
                [np.sin(phase), np.cos(phase)],
            ]
        ).astype(np.float32)
        with torch.inference_mode():
            self.previous = self.policy(torch.from_numpy(obs[None])).numpy()[0]
        action = np.zeros(env.model.nu, dtype=np.float32)
        action[:12] = self.previous * 0.25 / env.config.action_scale
        return np.clip(action, -1, 1)


def bootstrap(args):
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=False)
    config = EnvConfig(
        random_commands=True,
        episode_seconds=30,
        tracking_sigma=0.04,
        linear_velocity_observation=True,
        crouched_home=True,
        unitree_pd=True,
        phase_observation=True,
        action_scale=1.0,
        leg_only=True,
    )
    if args.data:
        with np.load(args.data) as data:
            observations, actions = data["observations"].copy(), data["actions"].copy()
    else:
        teacher = UnitreeTeacher(args.source)
        raw = G1Env(config)
        observations, actions = [], []
        obs, _ = raw.reset(seed=args.seed)
        try:
            for _ in range(args.samples):
                action = teacher.action(raw)
                observations.append(obs.copy())
                actions.append(action.copy())
                obs, _, terminated, truncated, _ = raw.step(action)
                if terminated or truncated:
                    teacher.reset()
                    obs, _ = raw.reset()
        finally:
            raw.close()
    observations = np.array(observations)
    actions = np.array(actions)
    np.savez_compressed(run / "teacher-data.npz", observations=observations, actions=actions)
    env = VecNormalize(DummyVecEnv([lambda: G1Env(config)]), norm_reward=True)
    env.obs_rms.update(observations)
    # Static upper-body/action channels otherwise amplify tiny prediction errors on rollout.
    env.obs_rms.var = np.maximum(env.obs_rms.var, 1e-4)
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=1024,
        batch_size=256,
        n_epochs=5,
        seed=args.seed,
        device="cpu",
        learning_rate=1e-5,
        target_kl=0.02,
        policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]}, "log_std_init": -3.5},
    )
    inputs = torch.from_numpy(env.normalize_obs(observations).astype(np.float32))
    targets = torch.from_numpy(actions)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=0.0003)
    losses = []
    try:
        for epoch in range(args.epochs):
            permutation = rng.permutation(len(inputs))
            epoch_loss = []
            for start in range(0, len(inputs), 256):
                index = permutation[start : start + 256]
                predicted = model.policy.get_distribution(inputs[index]).distribution.mean
                loss = torch.mean((predicted - targets[index]) ** 2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
            losses.append(float(np.mean(epoch_loss)))
            if (epoch + 1) % 10 == 0:
                print(f"imitation epoch {epoch + 1}: MSE={losses[-1]:.6f}", flush=True)
        model.bootstrap_origin = {
            "method": "behavior cloning from Unitree shipped policy",
            "source": "https://github.com/unitreerobotics/unitree_rl_gym",
            "commit": "276801e46c5d433564f24658bac64f254b7d2d4b",
            "not_from_scratch": True,
        }
        save_bundle(model, env, run / "final")
    finally:
        env.close()
    provenance = {
        "method": "behavior cloning initialization from Unitree shipped policy",
        "not_from_scratch": True,
        "teacher_robot_actions": 12,
        "student_robot_actions": 29,
        "teacher_source": "https://github.com/unitreerobotics/unitree_rl_gym",
        "teacher_commit": "276801e46c5d433564f24658bac64f254b7d2d4b",
        "samples": args.samples,
        "epochs": args.epochs,
        "seed": args.seed,
        "training_mse": losses,
        "env": config.to_dict(),
    }
    (run / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (run / "config.json").write_text(json.dumps({"env": config.to_dict(), "train": vars(args)}, indent=2))
    print(f"Initialized policy: {run / 'final'}")
