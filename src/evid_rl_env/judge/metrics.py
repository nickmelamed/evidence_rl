import numpy as np


def compute_ess(selected):
    return np.mean([1 if e.label == "support" else 0 for e in selected]) if selected else 0


def compute_ecs(selected):
    return np.mean([1 if e.label == "contradict" else 0 for e in selected]) if selected else 0


def compute_adversarial_penalty(selected):
    return np.mean([1 if e.label == "adversarial" else 0 for e in selected]) if selected else 0
