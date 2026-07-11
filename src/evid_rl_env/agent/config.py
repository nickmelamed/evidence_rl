class BaseConfig:
    def __init__(self):
        self.gamma = 0.99
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
        self.lr = 0.001
        self.clip = 0.2
        self.entropy_coef = 0.05  
        self.value_coef = 0.05
        self.gae_lambda = 0.95
        self.ppo_epochs = 2       

class PGConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.lr = 0.01
        self.lr_decay_episodes = 200   
        self.lr_min_fraction = 0.2    

class BanditConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.alpha = 1.0