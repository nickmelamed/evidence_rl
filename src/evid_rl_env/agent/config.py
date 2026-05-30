class BaseConfig:
    def __init__(self):
        self.gamma = 0.99
        self.actor_model = "google/gemma-2-2b-it"
        self.judge_model = "Qwen/Qwen2.5-1.5B-Instruct"
        self.seed = 42
        self.default_annotator_model = "claude-opus-4-5"
        self.eval_every = 10

class PPOConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.lr = 0.001
        self.clip = 0.2
        self.entropy_coef = 0.01
        self.value_coef = 0.05
        self.gae_lambda = 0.95
        self.ppo_epochs = 4

class PGConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.lr = 0.01

class BanditConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.alpha = 1.0