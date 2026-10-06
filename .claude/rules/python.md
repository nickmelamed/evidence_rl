---
paths:
  - "src/**/*.py"
  - "dashboard/**/*.py"
  - "configs/**"
---

# Python and configs

- Ruff is the linter (`select = E, F, I`, line length 100). E501 and E402 are off on purpose, since heavy imports are late.
- Never add `label` fields or evidence labels to what the agent observes, to prompts, or to the state encoder. Labels are for reward and evaluation only.
- Randomness: use the seeded paths already in the code. Do not call `random.seed` or `np.random.seed` at import time, and restore global RNG state after any temporary seeding, as the split helpers do.
- The train/eval split is the seed-42 80/20 shuffle. Do not write a new copy of it.
- Hyperparameters go in `configs/*.yaml`, not in the config classes.
- Tests must not call the network or load a model. Inject a mock judge, embedder, or labeler.
- Judge models used by `GoldEvaluator` must never be listed in `judge_model` or `judge_ensemble_models` of a training config.
