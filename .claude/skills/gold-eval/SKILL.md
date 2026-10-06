---
name: gold-eval
description: Re-score a checkpoint's held-out episodes with the gold judge and report proxy-vs-gold agreement. Only the owner starts this.
disable-model-invocation: true
---

Run the gold judge audit on one checkpoint. The owner names the checkpoint and the judge settings the run was trained with. If the checkpoint is missing, ask.

1. Read the run's `config.json` for `judge_model`, `judge_ensemble_models`, escalation settings, and seed.
2. Confirm the gold judge (`gold_judge_model` in `configs/base.yaml`) is not one of the training judges. Stop and tell the owner if it is.
3. Show the command and wait for approval, since it loads a 7B model:
   `evid-gold-eval --checkpoint <path> --judge-model <m> [--judge-ensemble-models ...] [--judge-escalation --judge-escalation-target <t>] --seed <seed>`
   The judge flags must match the training config, or the proxy scores are not the ones the policy saw.
4. Report `n_scored` out of `n_episodes`, `proxy_gold_correlation`, `outcome_accuracy`, `escalation_rate`, and the per-dimension disagreement. A `None` correlation means too few scored episodes or no variance, so say so rather than treating it as zero.
5. Note that gold eval samples stochastically (see D-004), so it is a reward audit and not a policy score.
