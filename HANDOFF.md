# Handoff: rerun the evaluations

This file is for a fresh conversation that will train and evaluate the models. Read CLAUDE.md, docs/SPEC.md and PROGRESS.md first. Do not push, merge, or change dependencies without asking the owner.

## Where things stand

- Branch `feat/llm-call-seeding` holds the latest work and may not be merged yet. Check `git log main..HEAD` and ask the owner before assuming.
- The code is ready to train. A real 10-episode PPO run plus an eval round ran end to end on this machine on 2026-10-06, and two identical runs gave identical rewards and weights, one with a cold judge cache and one with a warm one.
- Claim order, curriculum, policy init, baseline randomness and each LLM generation are seeded from `--seed` (default 42 from `configs/base.yaml`). Results from before this work cannot be compared with new ones.
- `gold_eval_n_episodes` is 100 in `configs/base.yaml`. The judge caches were emptied and archived in `artifacts/cache_archive/judge_caches_20261006/`. The evidence caches were kept.
- All needed models are in the local Hugging Face cache: gemma-2-2b-it, Qwen2.5-1.5B-Instruct, SmolLM2-1.7B-Instruct, Mistral-7B-Instruct-v0.2.

## Decide before starting

1. **Debate judge overlap (blocking for the debate run).** The debate arbiter is Mistral-7B-Instruct-v0.2, the same model as the default gold judge. For `ppo_debate_judge.yaml`, the gold judge is then not independent of the training reward. Ask the owner which to do: use a different gold judge for those runs, change the arbiter, or leave debate out of the gold comparison. Details in PROGRESS.md.
2. **Budget (TODO, not done).** Estimate time and disk before launching. A 10-episode run with one eval round took 13m47s cold and 3m42s warm. Old 300-episode runs took about a day each. Gold eval at 100 episodes is a large share of each run. Pick the architectures, seeds and episode count from that, and write the estimate into PROGRESS.md.
3. **Seeds.** One seed cannot support an architecture comparison. The owner may want 3 seeds per configuration. Seeds are passed with `--seed`.

## Commands

Run from the repo root with the venv active (`source ev_rl/bin/activate`). `TAVILY_API_KEY` must be set in `.env`. Never print `.env`.

Train one run per judge architecture (add `--seed N` for extra seeds, and use a distinct `--exp-name`):

```bash
evid-train --config configs/ppo_baseline.yaml --episodes 300 --exp-name full_baseline
evid-train --config configs/ppo_ensemble_judge.yaml --episodes 300 --exp-name full_ensemble
evid-train --config configs/ppo_escalation_judge.yaml --episodes 300 --exp-name full_escalation
evid-train --config configs/ppo_debate_judge.yaml --episodes 300 --exp-name full_debate   # after decision 1
```

Evaluate each checkpoint on the held-out split. The eval env uses the default single judge, so every architecture is scored the same way:

```bash
evid-eval --checkpoint artifacts/experiments/<run>/policy.npz \
  --baselines random,majority,greedy_llm,fewshot_k3,best_of_5 --n-episodes 100
```

Each training run writes `config.json` (with the device and library versions), `metrics.csv`, `gold_eval.jsonl` and `policy.npz` to `artifacts/experiments/<run>_<timestamp>/`.

## After the runs

1. Move the old experiment folders aside if the tables should show only the new runs. They currently include runs from before the seeding changes.
2. `make results` rebuilds `results/tables/` from `artifacts/experiments/`.
3. Update the README Results section from those tables. Quote only numbers that appear in them. `make numbers` fails until they match.
4. `make ci` must pass. Commit the tables and README together as `docs(results): ...`.
5. Update PROGRESS.md. Report failed or partial runs too, and do not drop rows that look bad.

## Things to watch

- Gold rounds still score few episodes if few rollouts reach `FINALIZE`. Report `n_scored` next to every correlation.
- Do not delete `artifacts/experiments/` folders or the cache archive without asking.
- Training is slow and model-heavy. Run it in the background, and check the logs rather than polling.
- If a run crashes, keep its folder and say so. Do not hide it.
- Rules that apply to every change are in CLAUDE.md. Protected paths need the owner's approval, including `configs/base.yaml`, `results/tables/` and `artifacts/experiments/`.
