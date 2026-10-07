# Handoff: rerun the evaluations

This file is for a fresh conversation that will train and evaluate the models. Read CLAUDE.md, docs/SPEC.md and PROGRESS.md first. Do not push, merge, delete experiment folders, or change dependencies without asking the owner.

## Where things stand

- All preparation is merged to `main`. Check `git status` and `git log -3` first.
- The code is ready to train. A real 10-episode PPO run plus an eval round ran end to end on 2026-10-06, and two identical runs gave identical rewards and weights, one with a cold judge cache and one with a warm one.
- Claim order, curriculum, policy init, baseline randomness and each LLM generation are seeded from `--seed`. Results from before this work cannot be compared with new ones.
- `gold_eval_n_episodes` is 100 in `configs/base.yaml`. The judge caches were emptied and archived in `artifacts/cache_archive/judge_caches_20261006/`. The evidence caches were kept.
- The debate config audits with `Qwen/Qwen2.5-3B-Instruct`, not Mistral-7B, because the debate arbiter is Mistral-7B (D-016). All other configs use Mistral-7B as the gold judge.
- Tavily has auto-reload on.
- All needed models are in the local Hugging Face cache: gemma-2-2b-it, Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct, SmolLM2-1.7B-Instruct, Mistral-7B-Instruct-v0.2.

## Plan

Four judge architectures times three seeds (42, 43, 44) is twelve runs. Start with seed 42 for all four and look at the results and timing before launching seeds 43 and 44. The owner said to adjust as needed, so cut seeds or episodes if one run takes too long. After the first run finishes, record its wall-clock time and disk use in PROGRESS.md under "Budget TODO".

Before the first new run, ask the owner to approve moving the old experiment folders to `artifacts/experiments_archive_pre_seeding/`. Otherwise the results tables mix old and new runs. `artifacts/experiments/` is a protected path.

## Commands

Run from the repo root with the venv active (`source ev_rl/bin/activate`). `TAVILY_API_KEY` must be set in `.env`. Never print `.env`.

Train, one run per architecture and seed. The `--exp-name` must be distinct:

```bash
S=42
evid-train --config configs/ppo_baseline.yaml --episodes 300 --seed $S --exp-name full_baseline_s$S
evid-train --config configs/ppo_ensemble_judge.yaml --episodes 300 --seed $S --exp-name full_ensemble_s$S
evid-train --config configs/ppo_escalation_judge.yaml --episodes 300 --seed $S --exp-name full_escalation_s$S
evid-train --config configs/ppo_debate_judge.yaml --episodes 300 --seed $S --exp-name full_debate_s$S
```

Evaluate each checkpoint on the held-out split. Use `--seed 42` for every run, whatever its training seed, so all runs are scored on identical episodes. The eval env uses the default single judge, so every architecture is scored the same way:

```bash
evid-eval --checkpoint artifacts/experiments/<run>/policy.npz --seed 42 \
  --baselines random,majority,greedy_llm,fewshot_k3,best_of_5 --n-episodes 100
```

Each training run writes `config.json` (with the device and library versions), `metrics.csv`, `gold_eval.jsonl` and `policy.npz` to `artifacts/experiments/<exp-name>_<timestamp>/`. `evid-eval` writes `eval_results.json` to the same folder.

Optional: rescore a checkpoint with a second gold judge, for example to compare debate with the others on one judge:
`evid-gold-eval --checkpoint <path> --gold-judge-model <model> ...` (match the judge flags to the training config).

## After the runs

1. `make results` rebuilds `results/tables/` from `artifacts/experiments/`. `summary_by_architecture.csv` gives the mean and spread across seeds.
2. Update the README Results section from those tables. Quote only numbers that appear in them. `make numbers` fails until they match.
3. `make ci` must pass. Commit the tables and README together as `docs(results): ...`.
4. Update PROGRESS.md. Report failed or partial runs too, and do not drop rows that look bad.

## Things to watch

- Gold rounds still score few episodes if few rollouts reach `FINALIZE`. Report `n_scored` next to every correlation.
- Debate gold agreement uses a different gold judge, so do not compare it directly with the others. Say so wherever it is reported.
- With three seeds the spread is a rough estimate. Report it as such.
- Do not delete `artifacts/experiments/` folders or the cache archive without asking.
- Training is slow and model-heavy. Run it in the background, and check the logs rather than polling.
- If a run crashes, keep its folder and say so. Do not hide it.
- Rules that apply to every change are in CLAUDE.md. Protected paths need the owner's approval, including `configs/base.yaml`, `results/tables/` and `artifacts/experiments/`.
