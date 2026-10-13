# Handoff

This file is for a fresh conversation. It has two tasks. Task 1 builds the multi-agent debate. Task 2 reruns the evaluations and is independent of Task 1. Read CLAUDE.md, docs/SPEC.md and PROGRESS.md first. Do not push, merge, delete experiment folders, or change dependencies without asking the owner.

# Task 1: multi-agent debate

Status on 2026-10-13: built on `feat/multi-agent-debate` and covered by mock tests (D-018, SPEC section 8, README). The gold audit of the pair and a debate token budget are also built. A real-model smoke run passed on 2026-10-13 (training, both gold paths, `eval_debate`). What is left is the owner's review and the arbiter question in PROGRESS.md (SmolLM2 often echoes the score template). The rest of this section is the original plan and decisions, kept for reference.

README Future Work item 2. Two independent agents argue the claim, one limited to support and one to contradiction, and a separate arbiter issues the reward. This separates role from policy, so a single agent no longer has to balance the debate itself.

The plan was approved on 2026-10-13 and the open questions are answered (see "Decisions" below). The full plan is in `~/.claude/plans/woolly-stargazing-metcalfe.md` on the owner's machine and is summarized here. Two things in it still need the owner's approval before code: the new `arbiter_model` config key, and the `DebateEvaluator`.

## Decisions

1. Reward split: the arbiter's score is shared by both roles.
2. Episode end: fixed turns, 5 per side, then the arbiter scores. `State.max_steps` stays 10.
3. Evidence: one shared pool and one shared selected set. `State` is unchanged.
4. Actor model: both policies share one `LLMClient`. The actor heads are numpy matrices, so the client is only used to write payload text. This avoids doubling memory and needs no `ActorCriticPolicy` redesign.
5. Arbiter model: `HuggingFaceTB/SmolLM2-1.7B-Instruct`, set by `arbiter_model` in YAML. The default Mistral-7B gold judge stays, so the config needs no D-016 exception.
6. Baselines: compare the debate pair with the single-agent PPO runs on the shared single judge, as `evid-eval` does.

## Already done on main

The PPO and PG updates were fixed on 2026-10-13, and this changes how the role masks are built. `ActorCriticPolicy.action_mask(state)` now builds the boolean mask, `masked_softmax(logits, mask)` in `agent/policy.py` is the one softmax used by `act`, `PPO.update` and `PolicyGradient.update`, and the trainer stores each step's features and mask. The update tuples are `(features, mask, action_idx, old_prob, reward, value)` for PPO and `(features, mask, action_idx, reward)` for PG. The role constraint is therefore a small layer on top: an optional `allowed_actions` argument to `action_mask` that removes `CONTRADICT` for the support role and `SUPPORT` for the contradict role. The masked PPO update test and the "never samples a masked action" test already exist in `tests/test_policy_masks.py`, so the debate tests only need to cover the role layer.

## What exists today

- `ClaimEnv` has one agent and 13 actions. `SUPPORT` and `CONTRADICT` both append to one `debate_history`, and the same agent chooses between them. The judge scores that history.
- `DebateJudge` (`judge/debate_judge.py`) is a different thing. Its two advocates argue whether the agent's reasoning is high or low quality, and its arbiter, Mistral-7B, scores the result. Its three models are fixed defaults and the YAML does not set them.
- `Trainer` (`agent/trainer.py`) runs one policy against one env, with PPO or PG updates per episode.
- `GoldEvaluator` and the evaluators are protected paths. The new work must not edit `agent/evaluator.py`, `agent/gold_evaluator.py`, `agent/eval_mixin.py`, `judge/reward.py` or `judge/metrics.py`.

## Proposed design

1. `environment/debate_env.py` adds `DebateEnv`, a wrapper over `ClaimEnv` that owns turn order. `ClaimEnv` stays unchanged, so existing runs and their results do not move. Each `DebateEnv.step` takes the role whose turn it is. The observation for a role is built from the same `State` summary the agents see today, so labels never enter it.
2. Role constraints are action masks. The support role may not use `CONTRADICT` and the contradict role may not use `SUPPORT`. Add an optional `allowed_actions` argument to `ActorCriticPolicy.action_mask` and `act`. The update already uses the stored mask, so no change to `PPO.update` is needed.
3. `agent/debate_trainer.py` adds `DebateTrainer`, built on `BaseTrainer`. It holds two policies and two PPO instances, keeps one trajectory per role, and updates each role from its own steps. It writes `policy_support.npz` and `policy_contradict.npz` to the run folder next to the usual `config.json`, `metrics.csv` and `gold_eval.jsonl`.
4. The arbiter is a new class in `judge/arbiter.py` that reuses `LLMJudge` scoring on the full two-sided transcript. The model name comes from `arbiter_model` in the config. The final reward goes to both roles. How it is split is open question 1.
5. `configs/ppo_multiagent_debate.yaml` sets both roles' hyperparameters, `arbiter_model` and `gold_judge_model`. The gold judge must differ from the arbiter. If the arbiter is not Mistral-7B, the default gold judge can stay as it is, so this config would not need the D-016 exception.
6. The existing guard test that no config's gold judge is a training judge must also read `arbiter_model`, and the test must fail first with a config that breaks the rule.
7. Evaluation needs a `DebateEvaluator` in a new file that plays the two trained policies on the same held-out episodes as every other method, with the same seed. This is a new evaluation path and touches the evaluation design, so ask the owner before writing it. The single-agent evaluator is not edited.
8. Docs: D-018 in `docs/DECISIONS.md` for the design and a SPEC section for the environment and trainer. Update the README, remove item 2 from Future Work, and keep every number in the README in `results/tables/`.

## Tests

- A role never samples an action outside its role mask. Use a Hypothesis test over random states and logits, plus a hand-worked example. The generic mask tests already exist.
- Turn order alternates and the episode ends after 5 turns per side.
- Two runs with the same seed give the same transcript and the same rewards. The global RNG state is restored after seeding.
- Observations and prompts for both roles contain no claim label and no evidence label.
- The arbiter test uses a mock judge. No test touches the network or loads a model.
- The gold-judge guard test described above.

## Order of work

1. Plan approved and questions answered (done).
2. Role layer on `action_mask` and `act`, with tests.
3. `DebateEnv` and the arbiter, with tests.
4. `DebateTrainer` and the config, with the guard test. Ask the owner about the `arbiter_model` key first.
5. `DebateEvaluator`, after the owner approves the evaluation change.
6. D-018, the SPEC section and the README.

Use a branch named `feat/multi-agent-debate`, cut from the current `main`, and Conventional Commits. A tiny end-to-end run of a few episodes with real models needs the owner's approval first, the same as any training run.

# Task 2: rerun the evaluations

This task trains and evaluates the models. It does not depend on Task 1, because Task 1 adds new files and leaves the existing environment, reward and evaluators unchanged.

## Where things stand

- All preparation is merged to `main`. Check `git status` and `git log -3` first. The rescore mode and the `AnnotatorError` change are also in, and neither touches training or evaluation.
- Two bugs in the PPO and PG updates were fixed and merged on 2026-10-13. `ClaimEnv` mutates one `State` in place, so every step used to be updated with the end-of-episode state, and the updaters used an unmasked softmax while `act` sampled from a masked one. Every PPO and PG run made before that is invalid, so the old experiment folders cannot be compared with new runs for that reason too. Bandit runs were not affected by those two. A third fix (2026-10-13) changes judge-scored rewards for every algorithm, bandit included: `SUPPORT` and `CONTRADICT` used to put the whole payload dict, with evidence ids and the token count, into the transcript the judge reads, and now put in only the argument text. It is merged to `main`, so the reruns can start.
- The code is ready to train. A real 10-episode PPO run plus an eval round ran end to end on 2026-10-06, and two identical runs gave identical rewards and weights, one with a cold judge cache and one with a warm one. That smoke test predates the 2026-10-13 fixes. Before the full runs, ask the owner whether to repeat a short two-run determinism check on the fixed code.
- Claim order, curriculum, policy init, baseline randomness and each LLM generation are seeded from `--seed`. Results from before this work cannot be compared with new ones.
- `gold_eval_n_episodes` is 100 in `configs/base.yaml`. The judge caches were emptied and archived in `artifacts/cache_archive/judge_caches_20261006/`. The evidence caches were kept.
- The debate config audits with `Qwen/Qwen2.5-3B-Instruct`, not Mistral-7B, because the debate arbiter is Mistral-7B (D-016). All other configs use Mistral-7B as the gold judge.
- Tavily has auto-reload on.
- All needed models are in the local Hugging Face cache: gemma-2-2b-it, Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct, SmolLM2-1.7B-Instruct, Mistral-7B-Instruct-v0.2.

## Plan

Four judge architectures times three seeds (42, 43, 44) is twelve runs. Start with seed 42 for all four and look at the results and timing before launching seeds 43 and 44. The owner said to adjust as needed, so cut seeds or episodes if one run takes too long. After the first run finishes, record its wall-clock time and disk use in PROGRESS.md under "Budget TODO".

The owner approved moving the old experiment folders on 2026-10-13, and all 13 are now in `artifacts/experiments_archive_pre_seeding/` (gitignored). `artifacts/experiments/` is empty, so `make results` will only see new runs. The committed tables in `results/tables/` still describe the old runs until the reruns rebuild them. `artifacts/experiments/` is a protected path.

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
