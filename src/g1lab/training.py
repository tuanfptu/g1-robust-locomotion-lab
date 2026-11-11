"""Low-memory PPO training with paired checkpoint directories and explicit resume."""

import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from .config import EnvConfig
from .env import G1Env


def make_env(config, seed, rank, monitor_dir=None):
    def factory():
        path = str(Path(monitor_dir) / str(rank)) if monitor_dir else None
        env = Monitor(G1Env(config), filename=path)
        env.reset(seed=seed + rank)
        return env

    return factory


def save_bundle(model, env, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    model.save(directory / "policy.zip")
    env.save(str(directory / "vecnormalize.pkl"))
    (directory / "bundle.json").write_text(json.dumps({"timesteps": model.num_timesteps}, indent=2))
    # Last file is a completion marker; incomplete bundles are never loaded.
    (directory / "COMPLETE").touch()


def check_bundle(directory):
    directory = Path(directory)
    if not all((directory / p).is_file() for p in ("policy.zip", "vecnormalize.pkl", "COMPLETE")):
        raise ValueError(f"Incomplete or missing checkpoint bundle: {directory}")
    return directory


class Checkpoints(BaseCallback):
    def __init__(self, run, interval):
        super().__init__()
        self.run, self.interval = Path(run), interval
        self.next_save = 0

    def _on_training_start(self):
        self.next_save = self.num_timesteps + self.interval

    def _on_step(self):
        if self.num_timesteps >= self.next_save:
            save_bundle(
                self.model, self.training_env, self.run / "checkpoints" / f"step-{self.num_timesteps}"
            )
            self.next_save += self.interval
        return True


def train(args):
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable. Use --device cpu; check NVIDIA driver before enabling CUDA.")
    config = EnvConfig(**json.loads(Path(args.config).read_text()))
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=False)
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except subprocess.CalledProcessError:
        revision = "uncommitted"
    metadata = {
        "env": config.to_dict(),
        "train": vars(args),
        "git_revision": revision,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    (run / "config.json").write_text(json.dumps(metadata, indent=2))
    (run / "monitor").mkdir()
    constructors = [make_env(config, args.seed, i, run / "monitor") for i in range(args.envs)]
    base = (
        SubprocVecEnv(constructors, start_method="spawn")
        if args.workers and args.envs > 1
        else DummyVecEnv(constructors)
    )
    if args.resume:
        checkpoint = check_bundle(args.resume)
        env = VecNormalize.load(str(checkpoint / "vecnormalize.pkl"), base)
        env.training, env.norm_reward = True, True
        model = PPO.load(checkpoint / "policy.zip", env=env, device=args.device)
    else:
        env = VecNormalize(base, norm_obs=True, norm_reward=True, clip_obs=10)
        batch_size = min(256, args.rollout * args.envs)
        model = PPO(
            "MlpPolicy",
            env,
            device=args.device,
            seed=args.seed,
            verbose=1,
            n_steps=args.rollout,
            batch_size=batch_size,
            n_epochs=5,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            policy_kwargs={"net_arch": dict(pi=[256, 256], vf=[256, 256])},
            tensorboard_log=str(run / "tensorboard"),
        )
    model.tensorboard_log = str(run / "tensorboard")
    start_steps, start = model.num_timesteps, time.perf_counter()
    interrupted = False
    try:
        model.learn(
            total_timesteps=args.steps,
            reset_num_timesteps=not bool(args.resume),
            callback=Checkpoints(run, args.checkpoint_every),
        )
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted: saving paired policy and normalization state.")
    finally:
        try:
            save_bundle(model, env, run / "final")
            elapsed = time.perf_counter() - start
            report = {
                "elapsed_seconds": elapsed,
                "additional_timesteps": model.num_timesteps - start_steps,
                "steps_per_second": (model.num_timesteps - start_steps) / elapsed,
                "interrupted": interrupted,
            }
            (run / "training.json").write_text(json.dumps(report, indent=2))
        finally:
            env.close()


def load_policy(bundle, config):
    checkpoint = check_bundle(bundle)
    raw = G1Env(config)
    vec = VecNormalize.load(str(checkpoint / "vecnormalize.pkl"), DummyVecEnv([lambda: raw]))
    vec.training, vec.norm_reward = False, False
    model = PPO.load(checkpoint / "policy.zip", device="cpu")
    return model, vec, raw


def policy_action(model, vec, observation):
    normalized = vec.normalize_obs(np.asarray(observation)[None])
    action, _ = model.predict(normalized, deterministic=True)
    return action[0]
