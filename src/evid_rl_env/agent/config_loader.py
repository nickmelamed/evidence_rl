import logging
from pathlib import Path

import yaml

from evid_rl_env.agent.config import BanditConfig, PGConfig, PPOConfig

logger = logging.getLogger(__name__)


def load_base_config(config_dir: str = "configs") -> dict:
    """Load base.yaml and return the raw dict. Returns {} if the file is absent."""
    path = Path(config_dir) / "base.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}

def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path: str):
    """Load base.yaml then deep-merge the algorithm yaml on top (algorithm wins).

    Expected algorithm YAML structure:
      algo: ppo          # or pg / bandit
      rl:
        lr: 0.001
        ...
    """
    config_dir = Path(path).parent
    with open(config_dir / "base.yaml") as f:
        base = yaml.safe_load(f) or {}
    with open(path) as f:
        algo_data = yaml.safe_load(f) or {}

    d = _deep_merge(base, algo_data)

    algo = d.get("algo", "ppo")
    if algo == "ppo":
        cfg = PPOConfig()
    elif algo == "pg":
        cfg = PGConfig()
    elif algo == "bandit":
        cfg = BanditConfig()
    else:
        raise ValueError(f"Unknown algo in config: {algo}")

    for k, v in d.get("rl", {}).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        else:
            logger.warning(
                "Config: rl.%s in %s has no matching field on %s — ignored. "
                "Add a matching attribute in agent/config.py if this should take effect.",
                k, path, type(cfg).__name__,
            )

    for k, v in d.items():
        if k in ("algo", "rl"):
            continue
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        else:
            logger.warning(
                "Config: top-level key %r in %s has no matching field on %s — ignored. "
                "Add a matching attribute in agent/config.py if this should take effect.",
                k, path, type(cfg).__name__,
            )

    logger.info("Config: base.yaml + %s.yaml | Seed: %s", Path(path).stem, cfg.seed)

    return algo, cfg
