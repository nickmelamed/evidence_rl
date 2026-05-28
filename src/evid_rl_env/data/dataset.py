import random

def load_dataset():
    return [
        {
            "claim": "AI models improve productivity.",
            "evidence": [
                {"id": 1, "text": "Study shows 20% increase", "label": "support"},
                {"id": 2, "text": "Some tasks degrade", "label": "contradict"},
                {"id": 3, "text": "Irrelevant blog", "label": "neutral"},
                {"id": 4, "text": "Misleading stat", "label": "adversarial"},
            ],
            "difficulty": "medium"
        }
    ]