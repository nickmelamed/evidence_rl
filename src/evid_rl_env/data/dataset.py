import json
import os
import random


def load_dataset():
    path = os.path.join(os.path.dirname(__file__), "seed_claims.json")
    with open(path) as f:
        return json.load(f)


SPLIT_SEED = 42
TRAIN_FRACTION = 0.8


def split_dataset(dataset: list) -> tuple[list, list]:
    """Return the fixed (train, eval) split used by training, eval and collection.

    The shuffle uses its own seed and puts the global RNG state back
    afterwards, so callers keep whatever seeding they set up.
    """
    saved_state = random.getstate()
    random.seed(SPLIT_SEED)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    random.setstate(saved_state)
    cut = int(TRAIN_FRACTION * len(dataset))
    return [dataset[i] for i in indices[:cut]], [dataset[i] for i in indices[cut:]]
