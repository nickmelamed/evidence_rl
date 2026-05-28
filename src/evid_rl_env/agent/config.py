class BaseConfig:
    def __init__(self):
        self.gamma = 0.99

class PPOConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.lr = 0.001
        self.clip = 0.2
        self.entropy_coef = 0.01
        self.value_coef = 0.05

class PGConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.lr = 0.01

class BanditConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.alpha = 1.0