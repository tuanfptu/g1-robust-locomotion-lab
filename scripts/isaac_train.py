"""Run the pinned Unitree RSL-RL train.py after registering g1lab task variants (no vendor edits).

Usage: python scripts/isaac_train.py <vendor>/scripts/rsl_rl/train.py [--critic_warmup N]
       [--critic_warmup_lr LR] [train.py arguments...]

--critic_warmup N runs the first N PPO iterations with the actor mean and action std frozen, so a value
function is fitted before policy updates start. Needed when the actor is transplanted from an ONNX export
that carries no critic. The N warm-up iterations count toward --max_iterations.
"""

import argparse
import runpy
import sys
from pathlib import Path

from g1lab.gpu import pin_cuda
from g1lab.isaac_tasks import register


def install_critic_warmup(iterations, learning_rate):
    from rsl_rl.runners import OnPolicyRunner

    original = OnPolicyRunner.learn

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        warmup = min(iterations, num_learning_iterations)
        policy = self.alg.policy
        frozen = [p for name, p in policy.named_parameters() if not name.startswith("critic.")]
        saved_lr = self.alg.learning_rate
        for p in frozen:
            p.requires_grad_(False)
        self.alg.learning_rate = learning_rate
        for group in self.alg.optimizer.param_groups:
            group["lr"] = learning_rate
        print(
            f"[g1lab] critic warm-up: {warmup} iterations, {len(frozen)} actor tensors frozen, lr={learning_rate}"
        )
        original(self, warmup, init_at_random_ep_len)
        for p in frozen:
            p.requires_grad_(True)
        self.alg.learning_rate = saved_lr
        for group in self.alg.optimizer.param_groups:
            group["lr"] = saved_lr
        self.current_learning_iteration += 1
        print(f"[g1lab] critic warm-up finished; actor unfrozen, lr={saved_lr}")
        if num_learning_iterations > warmup:
            original(self, num_learning_iterations - warmup, False)

    OnPolicyRunner.learn = learn


if __name__ == "__main__":
    pin_cuda()
    train = Path(sys.argv[1]).resolve()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--critic_warmup", type=int, default=0)
    parser.add_argument("--critic_warmup_lr", type=float, default=1.0e-3)
    ours, rest = parser.parse_known_args(sys.argv[2:])
    register()
    if ours.critic_warmup > 0:
        install_critic_warmup(ours.critic_warmup, ours.critic_warmup_lr)
    sys.argv = [str(train), *rest]
    sys.path.insert(0, str(train.parent))  # what `python train.py` would provide for its local imports
    runpy.run_path(str(train), run_name="__main__")
