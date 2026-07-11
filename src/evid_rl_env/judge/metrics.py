_RELEVANT_LABELS = ("support", "contradict")


def compute_precision(selected):
    if not selected:
        return 0.0
    return sum(1 for e in selected if e.label in _RELEVANT_LABELS) / len(selected)


def compute_recall(selected, available):
    support_available = [e for e in available if e.label == "support"]
    if not support_available:
        return 1.0
    return sum(1 for e in support_available if e in selected) / len(support_available)


def compute_f1(selected, available):
    precision = compute_precision(selected)
    recall = compute_recall(selected, available)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_contradiction_acknowledgment(selected, available):
    contradict_available = [e for e in available if e.label == "contradict"]
    if not contradict_available:
        return 1.0
    return sum(1 for e in contradict_available if e in selected) / len(contradict_available)


def compute_adversarial_contamination(selected):
    if not selected:
        return 0.0
    return sum(1 for e in selected if e.label == "adversarial") / len(selected)
