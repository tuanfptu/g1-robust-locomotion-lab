"""Low-memory PPO training with paired checkpoint directories and explicit resume."""

import json
import platform
import subprocess
import time
from dataclasses import replace
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
    config = env.get_attr("config")[0]
    (directory / "env_config.json").write_text(json.dumps(config.to_dict(), indent=2))
    (directory / "environment_mix.json").write_text(
        json.dumps([item.to_dict() for item in env.get_attr("config")], indent=2)
    )
    if hasattr(model, "bootstrap_origin"):
        (directory / "origin.json").write_text(json.dumps(model.bootstrap_origin, indent=2))
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


class CriticWarmup(BaseCallback):
    def __init__(self, steps):
        super().__init__()
        self.steps = steps

    def _set_actor(self, enabled):
        policy = self.model.policy
        for module in (policy.mlp_extractor.policy_net, policy.action_net):
            for parameter in module.parameters():
                parameter.requires_grad_(enabled)
        policy.log_std.requires_grad_(enabled)

    def _on_training_start(self):
        self.until = self.num_timesteps + self.steps
        self._set_actor(False)

    def _on_step(self):
        if self.num_timesteps >= self.until:
            self._set_actor(True)
        return True

    def _on_training_end(self):
        self._set_actor(True)


class BehaviorAnchor(BaseCallback):
    """Explicit behavioral replay between PPO rollouts; source is recorded in run config."""

    def __init__(self, path):
        super().__init__()
        with np.load(path) as data:
            self.observations = data["observations"].copy()
            self.actions = data["actions"].copy()
        self.rng = np.random.default_rng(123)

    def _on_training_start(self):
        self.optimizer = torch.optim.Adam(self.model.policy.parameters(), lr=0.0001)

    def _anchor(self):
        if not next(self.model.policy.action_net.parameters()).requires_grad:
            return
        for _ in range(32):
            index = self.rng.integers(len(self.observations), size=256)
            inputs = torch.as_tensor(
                self.training_env.normalize_obs(self.observations[index]),
                dtype=torch.float32,
                device=self.model.device,
            )
            targets = torch.as_tensor(self.actions[index], dtype=torch.float32, device=self.model.device)
            predicted = self.model.policy.get_distribution(inputs).distribution.mean
            loss = ((predicted[:, :12] - targets[:, :12]) ** 2).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def _on_rollout_start(self):
        self._anchor()

    def _on_step(self):
        return True

    def _on_training_end(self):
        self._anchor()


def train(args):
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable. Use --device cpu; check NVIDIA driver before enabling CUDA.")
    config_paths = [args.config, *getattr(args, "mix_config", [])]
    configs = [EnvConfig(**json.loads(Path(path).read_text())) for path in config_paths]
    config = configs[0]
    if len(configs) > args.envs:
        raise ValueError("Use at least one environment per mixed configuration")
    contract = (
        "task",
        "control_dt",
        "physics_dt",
        "linear_velocity_observation",
        "action_scale",
        "crouched_home",
        "unitree_pd",
        "phase_observation",
        "leg_only",
    )
    if any(any(getattr(other, key) != getattr(config, key) for key in contract) for other in configs):
        raise ValueError("Mixed environments must share the policy observation/action contract")
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=False)
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
            ).strip()
        )
    except subprocess.CalledProcessError:
        revision = "uncommitted"
        dirty = True
    metadata = {
        "env": config.to_dict(),
        "environment_mix": [configs[i % len(configs)].to_dict() for i in range(args.envs)],
        "train": vars(args),
        "git_revision": revision,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    (run / "config.json").write_text(json.dumps(metadata, indent=2))
    (run / "monitor").mkdir()
    constructors = [
        make_env(configs[i % len(configs)], args.seed, i, run / "monitor") for i in range(args.envs)
    ]
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
        if (checkpoint / "origin.json").exists():
            model.bootstrap_origin = json.loads((checkpoint / "origin.json").read_text())
        model.learning_rate = args.learning_rate
        model.lr_schedule = lambda _: args.learning_rate
        model.target_kl = args.target_kl
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
            learning_rate=args.learning_rate,
            target_kl=args.target_kl,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            policy_kwargs={"net_arch": dict(pi=[256, 256], vf=[256, 256]), "log_std_init": args.log_std_init},
            tensorboard_log=str(run / "tensorboard"),
        )
    model.tensorboard_log = str(run / "tensorboard")
    if args.freeze_normalization:
        env.training = False
        env.norm_reward = False
    metadata["effective_ppo"] = {
        "n_steps": model.n_steps,
        "n_envs": env.num_envs,
        "batch_size": model.batch_size,
        "n_epochs": model.n_epochs,
        "initial_timesteps": model.num_timesteps,
        "normalization_training": env.training,
        "normalize_reward": env.norm_reward,
    }
    (run / "config.json").write_text(json.dumps(metadata, indent=2))
    start_steps, start = model.num_timesteps, time.perf_counter()
    interrupted = False
    try:
        callbacks = [Checkpoints(run, args.checkpoint_every)]
        if args.freeze_actor_steps > 0:
            callbacks.insert(0, CriticWarmup(args.freeze_actor_steps))
        if args.anchor_data:
            callbacks.append(BehaviorAnchor(args.anchor_data))
        model.learn(
            total_timesteps=args.steps,
            reset_num_timesteps=not bool(args.resume),
            callback=callbacks,
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
    specification = checkpoint / "env_config.json"
    if specification.exists():
        contract = json.loads(specification.read_text())
        fields = (
            "task",
            "control_dt",
            "physics_dt",
            "linear_velocity_observation",
            "action_scale",
            "crouched_home",
            "unitree_pd",
            "phase_observation",
            "leg_only",
        )
        config = replace(config, **{key: contract[key] for key in fields if key in contract})
    raw = G1Env(config)
    vec = VecNormalize.load(str(checkpoint / "vecnormalize.pkl"), DummyVecEnv([lambda: raw]))
    vec.training, vec.norm_reward = False, False
    model = PPO.load(checkpoint / "policy.zip", device="cpu")
    return model, vec, raw


def policy_action(model, vec, observation):
    normalized = vec.normalize_obs(np.asarray(observation)[None])
    action, _ = model.predict(normalized, deterministic=True)
    return action[0]
