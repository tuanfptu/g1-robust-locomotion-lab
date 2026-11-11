"""Train and evaluate a MuJoCo humanoid with paired policy/normalization files."""
import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def train(args):
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    if (run / "config.json").exists():
        raise SystemExit("Run already exists. Choose a new --run directory.")
    (run / "config.json").write_text(json.dumps(vars(args), indent=2))
    env = VecNormalize(make_vec_env("Humanoid-v5", n_envs=args.envs,
                                   seed=args.seed, monitor_dir=str(run / "monitor")))
    model = PPO("MlpPolicy", env, seed=args.seed, device="cpu", verbose=1,
                n_steps=2048, batch_size=256, learning_rate=3e-4,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                policy_kwargs={"net_arch": dict(pi=[256, 256], vf=[256, 256])},
                tensorboard_log=str(run / "tensorboard"))
    checkpoint = CheckpointCallback(max(100_000 // args.envs, 1),
                                    str(run / "checkpoints"),
                                    name_prefix="ppo", save_vecnormalize=True)
    try:
        model.learn(total_timesteps=args.steps, callback=checkpoint)
        model.save(run / "policy")
        env.save(str(run / "vecnormalize.pkl"))
    finally:
        env.close()


def evaluate(args):
    run = Path(args.run)
    base = gym.make("Humanoid-v5", render_mode="rgb_array" if args.video else None)
    if args.video:
        base = gym.wrappers.RecordVideo(base, str(run / "videos"),
                                       episode_trigger=lambda episode: True)
    env = VecNormalize.load(str(run / "vecnormalize.pkl"), DummyVecEnv([lambda: base]))
    env.training = False
    env.norm_reward = False
    env.seed(args.seed)
    model = PPO.load(run / "policy", device="cpu")
    episodes = []
    try:
        obs = env.reset()
        total, length = 0.0, 0
        while len(episodes) < args.episodes:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = env.step(action)
            total += float(reward[0])
            length += 1
            if done[0]:
                episodes.append({"return": total, "length": length,
                                 "x_position": float(infos[0].get("x_position", 0))})
                total, length = 0.0, 0
        report = {"seed": args.seed, "episodes": episodes,
                  "mean_return": float(np.mean([e["return"] for e in episodes])),
                  "std_return": float(np.std([e["return"] for e in episodes])),
                  "mean_length": float(np.mean([e["length"] for e in episodes]))}
        (run / "evaluation.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    finally:
        env.close()


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "eval"])
    parser.add_argument("--run", default="runs/humanoid-ppo-seed42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=positive, default=3_000_000)
    parser.add_argument("--envs", type=positive, default=4)
    parser.add_argument("--episodes", type=positive, default=10)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    (train if args.mode == "train" else evaluate)(args)


if __name__ == "__main__":
    main()
