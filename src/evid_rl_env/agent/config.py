class BaseConfig:
    """Fields shared across every algorithm, safe as real hardcoded defaults
    because they aren't per-run tuning knobs (model choices, seed, and other
    cross-cutting operational defaults sourced from configs/base.yaml).

    Algorithm-specific RL hyperparameters live on the subclasses below and are
    intentionally left as None — configs/*_baseline.yaml's `rl:` block is the
    single source of truth for those; see config_loader.load_config(), which
    always overwrites them before a config is used for real training.
    """
    def __init__(self):
        self.actor_model = "google/gemma-2-2b-it"
        self.judge_model = "Qwen/Qwen2.5-1.5B-Instruct"
        self.seed = 42
        self.default_annotator_model = "claude-opus-4-5"
        self.eval_every = 10
        self.max_grad_norm = 0.5
        # "random" or "similarity" (sentence-transformers embedding lookup) —
        # selects FewShotLLMBaseline's few-shot example retrieval strategy.
        self.fewshot_selection_mode = "random"

class PPOConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.lr = None
        self.clip = None
        self.gamma = None
        self.entropy_coef = None
        self.value_coef = None
        self.gae_lambda = None
        self.ppo_epochs = None

class PGConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.lr = None
        self.lr_decay_episodes = None
        self.lr_min_fraction = None

class BanditConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.alpha = None