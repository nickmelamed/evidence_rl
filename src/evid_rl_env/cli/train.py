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


def train(episodes, method="ppo", config_path=None):
    dataset = load_dataset()
    curriculum = Curriculum()

    sampled_dataset = [curriculum.sample(dataset) for _ in range(len(dataset))]
    env = ClaimEnv(sampled_dataset)

    policy = ActorCriticPolicy(len(list(ACTIONS)))

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
            exp_name=f"{method}_run"
        )
    else:
        trainer = Trainer(
            env=env,
            policy=policy,
            config=config,
            episodes=episodes,
            algo=method,
            exp_name=f"{method}_run"
        )

    trainer.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--method", type=str, default="ppo")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file (e.g. configs/ppo_baseline.yaml)")
    args = parser.parse_args()

    train(args.episodes, args.method, args.config)


if __name__ == "__main__":
    main()