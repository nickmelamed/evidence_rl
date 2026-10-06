---
name: run-experiment
description: Train one method from a config, then evaluate its checkpoint. Only the owner starts this.
disable-model-invocation: true
---

Run one training experiment. The owner names the config (for example `configs/ppo_baseline.yaml`) and the episode count. If either is missing, ask.

1. Confirm the working tree is clean and `TAVILY_API_KEY` is set in `.env` without reading its value. Check that the config's judge models do not include `gold_judge_model` from `configs/base.yaml`.
2. Show the exact command and wait for approval, since training loads models and can run for hours:
   `evid-train --config <config> --method <ppo|pg|bandit> --episodes <n> --seed <seed>`
3. Run it. Report the run folder under `artifacts/experiments/`, the final reward, and any warnings from the log.
4. Evaluate on the held-out split with the same seed:
   `evid-eval --checkpoint artifacts/experiments/<run>/policy.npz --baselines random,greedy_llm,fewshot_k3,best_of_5 --n-episodes 50 --seed <seed>`
5. Show the printed table. Do not copy numbers into the README. Use the results-table skill for that.
6. Never delete or edit an existing run folder.
