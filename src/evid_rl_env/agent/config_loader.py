import yaml
from evid_rl_env.agent.config import PPOConfig, PGConfig, BanditConfig


def load_config(path: str):
    """Load a config from a YAML file and return the appropriate config object.

    Expected YAML structure:
      algo: ppo          # or pg / bandit
      rl:
        lr: 0.001
        clip: 0.2
        entropy_coef: 0.01
        value_coef: 0.05
        gamma: 0.99
        ppo_epochs: 4    # PPO only
        alpha: 1.0       # Bandit only
    """
    with open(path) as f:
        d = yaml.safe_load(f)

    algo = d.get("algo", "ppo")
    rl = d.get("rl", {})

    if algo == "ppo":
        cfg = PPOConfig()
    elif algo == "pg":
        cfg = PGConfig()
    elif algo == "bandit":
        cfg = BanditConfig()
    else:
        raise ValueError(f"Unknown algo in config: {algo}")

    for k, v in rl.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    return algo, cfg
