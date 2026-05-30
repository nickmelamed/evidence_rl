from evid_rl_env.environment.environment import ClaimEnv
from evid_rl_env.environment.curriculum import Curriculum
from evid_rl_env.agent.policy import ActorCriticPolicy
from evid_rl_env.environment.actions import ACTIONS
from evid_rl_env.data.dataset import load_dataset
from evid_rl_env.agent.trainer import Trainer
from evid_rl_env.agent.bandit_trainer import BanditTrainer
from evid_rl_env.agent.config import PPOConfig, PGConfig, BanditConfig
from evid_rl_env.agent.config_loader import load_config

import argparse
import random


def train(episodes, method="ppo", config_path=None, seed=42):
    dataset = load_dataset()

    # AUDIT FIX: isolate the split seed from training randomness — seed 42 must always
    # match eval.py's _deterministic_split so train/eval sets are identical across runs;
    # save/restore so this doesn't pollute the seed that Trainer uses for weight init
    _saved_state = random.getstate()
    random.seed(42)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    split = int(0.8 * len(dataset))
    train_dataset = [dataset[i] for i in indices[:split]]
    eval_dataset = [dataset[i] for i in indices[split:]]
    random.setstate(_saved_state)

    curriculum = Curriculum()
    sampled_dataset = [curriculum.sample(train_dataset) for _ in range(len(train_dataset))]
    # AUDIT FIX: pass seed so ClaimEnv threads it to JudgeLLMClient for reproducible
    # judge outputs and ActorCriticPolicy threads it to LLMClient for reproducible actor outputs
    env = ClaimEnv(sampled_dataset, seed=seed)

    policy = ActorCriticPolicy(len(list(ACTIONS)), seed=seed)

    if config_path:
        method, config = load_config(config_path)
    elif method == 'ppo':
        config = PPOConfig()
    elif method == 'pg':
        config = PGConfig()
    elif method == 'bandit':
        config = BanditConfig()

    if method == 'bandit':
        trainer = BanditTrainer(
            env=env,
            policy=policy,
            config=config,
            episodes=episodes,
            exp_name=f"{method}_run",
            seed=seed,
            # AUDIT FIX: pass eval_dataset so BanditTrainer can build an isolated eval env
            eval_dataset=eval_dataset,
        )
    else:
        trainer = Trainer(
            env=env,
            policy=policy,
            config=config,
            episodes=episodes,
            algo=method,
            exp_name=f"{method}_run",
            eval_dataset=eval_dataset,
            seed=seed,
        )

    trainer.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--method", type=str, default="ppo")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file (e.g. configs/ppo_baseline.yaml)")
    # AUDIT FIX: expose --seed so training runs are reproducible; consistent with
    # --seed in evid-eval and evid-collect
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for weight initialisation and action sampling (default: 42). "
             "The train/eval split always uses seed 42 regardless of this value.",
    )
    args = parser.parse_args()

    train(args.episodes, args.method, args.config, seed=args.seed)


if __name__ == "__main__":
    main()