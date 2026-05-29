import json
import os


def load_dataset():
    path = os.path.join(os.path.dirname(__file__), "seed_claims.json")
    with open(path) as f:
        return json.load(f)
