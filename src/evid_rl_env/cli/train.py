import faulthandler
import os

faulthandler.enable()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import random

from evid_rl_env.agent.bandit_trainer import BanditTrainer
from evid_rl_env.agent.config_loader import load_base_config, load_config
from evid_rl_env.agent.policy import ActorCriticPolicy
from evid_rl_env.agent.trainer import Trainer
from evid_rl_env.data.dataset import load_dataset
from evid_rl_env.data.evidence_fetcher import use_snapshot, warm_cache
from evid_rl_env.environment.actions import ACTIONS
from evid_rl_env.environment.curriculum import Curriculum
from evid_rl_env.environment.environment import ClaimEnv

# configs/*_baseline.yaml is the single source of truth for RL hyperparameters
# (see agent/config.py) — even the no-"--config" default path goes through
# load_config() against these files rather than bare-constructing a Config
# class, so a run never silently ends up with unset (None) tuning values.
_DEFAULT_CONFIG_PATHS = {
    "ppo": "configs/ppo_baseline.yaml",
    "pg": "configs/pg_baseline.yaml",
    "bandit": "configs/bandit_baseline.yaml",
}


def train(episodes, method="ppo", config_path=None, seed=42, eval_every=None, evidence_snapshot=None):
    if evidence_snapshot is not None:
        use_snapshot(evidence_snapshot)

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

    warm_cache(train_dataset + eval_dataset)

    resolved_config_path = config_path or _DEFAULT_CONFIG_PATHS.get(method)
    if resolved_config_path is None:
        raise ValueError(f"Unknown method: {method}")
    method, config = load_config(resolved_config_path)
    config.seed = seed  # align config.seed with the resolved seed

    curriculum = Curriculum()
    # Config loaded first so judge_model/actor_model actually drive which
    # models get instantiated instead of falling back to hardcoded defaults.
    env = ClaimEnv(train_dataset, judge_model=config.judge_model, seed=seed)

    policy = ActorCriticPolicy(len(list(ACTIONS)), model_name=config.actor_model, seed=seed)

    resolved_eval_every = eval_every if eval_every is not None else getattr(config, "eval_every", 25)

    if method == 'bandit':
        trainer = BanditTrainer(
            env=env,
            policy=policy,
            config=config,
            episodes=episodes,
            exp_name=f"{method}_run",
            seed=seed,
            eval_dataset=eval_dataset,
            eval_every=resolved_eval_every,
            curriculum=curriculum,
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
            eval_every=resolved_eval_every,
            curriculum=curriculum,
        )

    trainer.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--method", type=str, default="ppo")
    parser.add_argument("--eval-every", type=int, default=None,
                        help="Run evaluation every N episodes. Defaults to configs/base.yaml's "
                             "eval_every (currently 10).")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file (e.g. configs/ppo_baseline.yaml)")
    parser.add_argument(
        "--evidence-snapshot",
        type=str,
        default=None,
        help="Path to a JSON snapshot from evid-snapshot. If set, evidence for "
             "matching claims is reproduced from the snapshot instead of the "
             "live Tavily/sqlite-cache path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for weight initialisation and action sampling. "
             "Defaults to the value in configs/base.yaml (currently 42). "
             "The train/eval split always uses seed 42 regardless of this value.",
    )
    args = parser.parse_args()

    if args.seed is not None:
        seed = args.seed
    else:
        seed = load_base_config().get("seed", 42)

    train(args.episodes, args.method, args.config, seed=seed,
          eval_every=args.eval_every, evidence_snapshot=args.evidence_snapshot)


if __name__ == "__main__":
    main()