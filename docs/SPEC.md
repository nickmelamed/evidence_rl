# EvidenceRL spec (DRAFT)

Marks: **[confirmed]** means the owner stated it. **[inferred]** means it was read from the code or README and has not been confirmed. Correct anything marked inferred.

## 1. Purpose

EvidenceRL treats claim verification as a sequential decision problem. An agent works in a Gym-style environment (`ClaimEnv`), gathers and argues over evidence for a scientific claim, and is rewarded by a mix of heuristic evidence metrics and an LLM judge. The project has two parts: the RL framework (bandit, REINFORCE, PPO) and a study of whether LLM-judge reward architectures (single, debate, ensemble, escalating) track an independent gold judge. **[confirmed]**

## 2. Rules that must not be broken

1. **No train/eval leakage.** Train and eval sets come from one deterministic seed-42 80/20 shuffle of `seed_claims.json`. Eval claims are never used for training, the curriculum, few-shot examples, or imitation data. **[confirmed]**
2. **Gold judges stay out of training reward.** Judges used by `GoldEvaluator` never contribute to the reward the policy trains on. Otherwise proxy-vs-gold agreement means nothing. **[confirmed]**
3. **Reported numbers come from real runs.** Numbers in the README or a write-up trace to a run in `artifacts/experiments/` or a committed results table. Nothing is typed in by hand. **[confirmed]**
4. **Randomness is seeded.** The same seed gives the same result. Evaluation across methods uses the same episodes. **[confirmed]**
5. **Labels never reach the agent.** The claim `label` and evidence `label` fields are used only for reward and evaluation, never in the observation or the agent's prompts. **[confirmed]**
6. **Secrets stay out of git.** `.env` is never tracked. `.env.example` holds names only. **[inferred]**

## 3. Data

- `src/evid_rl_env/data/seed_claims.json`: 693 SciFact claims, `label` 1.0 for SUPPORT and 0.0 for CONTRADICT, with gold evidence abstracts. CC BY-NC, see `ATTRIBUTION.md`. **[inferred]**
- Claims that carry gold `evidence` use it directly with its human labels. Claims without it fetch live Tavily results and label each document with `EvidenceLabeler` (an LLM call). **[inferred]**
- `--evidence-snapshot` loads a portable JSON snapshot of fetched evidence instead of calling Tavily. **[inferred, from README]**
- Caches (`artifacts/cache/*.sqlite3`) hold fetch, label and judge results. They are gitignored. **[inferred]**

## 4. Environment (`ClaimEnv`)

- Thirteen actions (`environment/actions.py`): `SELECT`, `REMOVE`, `SUPPORT`, `CONTRADICT`, `FINALIZE`, `QUERY`, `RERANK`, `SUMMARIZE`, `CONCEDE`, `ASSIGN_CONFIDENCE`, `CHALLENGE_EVIDENCE`, `REQUEST_CLARIFICATION`, `HEDGE`. **[inferred]**
- `QUERY` and `REQUEST_CLARIFICATION` share a budget of `max_queries` per episode (README says 2). **[inferred]**
- The judge is called at most once every two steps. In between, the previous score is reused. **[inferred]**
- Episodes end on `FINALIZE` with at least one selected evidence item, or at the step limit. The README says 10 and the limit lives in `State.is_done`. **[inferred]**
- `FINALIZE` with no evidence ends the episode at -1.0. `FINALIZE` at step 2 or earlier gives -0.5 and the episode continues. **[inferred]**
- Every step reward gets `0.1 * (0.99 * phi_next - phi_prev)`, where phi is the last judge score and is 0 at terminal states. **[inferred]**
- `task_success = 1 - |confidence - label|` is set only on `FINALIZE`. The curriculum and gold eval use it. **[inferred]**
- `confidence` is the agent's `ASSIGN_CONFIDENCE` value, or `min(1, n_selected / 3)` if it never called it. **[inferred]**
- `SUPPORT` and `CONTRADICT` add `"SUPPORT: "` or `"CONTRADICT: "` plus the payload's `argument` text to the transcript the judge reads. The evidence ids and token count in the payload are not written. **[confirmed]**

## 5. Reward

- Step rewards are defined in `ClaimEnv.step`.
- Final reward is `RewardFunction.compute`: `0.40*F1 + 0.20*CA - 0.15*AC - 0.15*|confidence - label| + 0.10*llm_reward - 0.04*max(0, n_selected - 5)`, clipped to [-1, 1]. Empty reasoning gives 0.0. Then `-0.3` if the debate history is empty, and `+min(0.3, 0.1 * useful_selected)`. **[inferred]**
- Judge score to reward: `0.30*LCS + 0.25*ESS + 0.20*COMP - 0.25*GRS - 0.15*BIAS`, blended toward 0.5 by `(1 - confidence)`, clipped to [0, 1]. An all-0.5 score is treated as parse failure and returns 0.5. **[inferred]**
- Metrics (`judge/metrics.py`): precision, recall, F1, contradiction acknowledgment (CA), and adversarial contamination (AC). **[inferred]**

## 6. Judges

`LLMJudge` is the single judge. `EnsembleJudge` aggregates several models. `DebateJudge` uses an arbiter. `EscalatingJudge` runs a cheap judge first and escalates to the ensemble or debate judge. Judge scores are cached by an md5 of claim, reasoning and evidence text. Parse failures and generation failures return neutral scores (the latter are not cached). **[inferred]**

## 7. Training and evaluation

- Trainers: bandit, REINFORCE, PPO. Hyperparameters live in `configs/*.yaml`, and the YAML is the single source of truth. **[inferred]**
- The curriculum is per-claim Prioritized Level Replay over the train split. **[inferred]**
- Evaluation (`evid-eval`): the RL policy against `random`, `majority`, `greedy_llm`, `fewshot_k3`, `fewshot_k5`, `best_of_5` and `imitation`, on the held-out split. The README shows mean ± std reward with the delta against `greedy_llm`. **[confirmed]**
- Gold eval (`evid-gold-eval`) re-scores final reasoning with held-out judges (default `mistralai/Mistral-7B-Instruct-v0.2`). It reports `proxy_gold_correlation`, `outcome_accuracy` and per-dimension disagreement. **[inferred]**
- Results are committed as tables in `results/tables/`, built by `scripts/build_results_tables.py` from `artifacts/experiments/`. The README quotes only those tables, and `make numbers` checks it. The Streamlit dashboard still reads the run folders directly. **[confirmed]**

## 8. Multi-agent debate

- `DebateEnv` (`environment/debate_env.py`) wraps a `ClaimEnv`. Two roles alternate, support first, for 5 turns each. The environment then runs `FINALIZE` itself, so the arbiter scores the whole transcript. The step limit is 11 for these episodes. **[confirmed]**
- The roles share one evidence pool, one selected set and one transcript. `State` is unchanged. **[confirmed]**
- The support role may not use `CONTRADICT`, the contradict role may not use `SUPPORT`, and neither may use `FINALIZE`. On the last turn, if no confidence has been assigned and evidence is selected, the only allowed action is `ASSIGN_CONFIDENCE`. The evidence-count fallback in `FINALIZE` remains as a backstop. **[confirmed]**
- `Arbiter` (`judge/arbiter.py`) wraps an `LLMJudge` over `arbiter_model` with its own score cache and is passed to the env as `llm_judge`. The reward formula is unchanged. **[confirmed]**
- `DebateTrainer` (`agent/debate_trainer.py`) holds two policies and two PPO instances. Each role updates from its own turns. The final reward is added to each role's last step. It writes `policy_support.npz` and `policy_contradict.npz`, not `policy.npz`. **[confirmed]**
- `max_tokens_per_episode` (6000 in the config) ends a debate early when the roles' payloads reach it. The debate is still finalized and scored, and `info["truncated"]` is set. **[confirmed]**
- Gold audit: `build_debate_gold_evaluator` (`agent/debate_gold.py`) runs the unchanged `GoldEvaluator` on the pair. The proxy side is the arbiter's final score, and the gold judge re-scores the same two-sided transcript. `DebateTrainer` runs it every `eval_every * gold_eval_every` episodes when given an eval set and appends to `gold_eval.jsonl`, and `eval_debate --gold-judge-model` runs it on a saved run. Both refuse a gold judge that is the arbiter. **[confirmed]**
- `configs/ppo_multiagent_debate.yaml` (`algo: ppo_multiagent`) sets `arbiter_model`, `turns_per_side` and the PPO hyperparameters. `evid-train --config` routes it to the debate trainer. **[confirmed]**
- `DebateEvaluator` (`agent/debate_evaluator.py`) and `python -m evid_rl_env.cli.eval_debate --run-dir <run>` play the pair greedily on the held-out split with the same single-judge env and seed as `evid-eval`. An episode's reward is the sum of both roles' step rewards plus the final reward. **[confirmed]**

## 9. Known gaps

- The seed-42 split lives in `data/dataset.py` (`split_dataset`) and is used by train, eval and collect. The collect copy failed to restore global RNG state until commit 1d98db8, which made `evid-collect --seed` ineffective before then.
- `ClaimEnv.reset` samples claims from its own `random.Random(seed)`. `Evaluator` and `GoldEvaluator` call `reseed()` at the start of each round, and baselines are built with the evaluation seed, so every method sees the same claims in the same order.
- `train.py` gives the curriculum its own `random.Random(seed)` and seeds numpy before the policy draws its initial weights. Baselines are reseeded before each run in `evid-eval` and in the in-training eval rounds, so a baseline's random actions do not depend on which baselines ran before it.
- Model sampling is not fully deterministic. The actor and judge models run through `transformers`, and on Apple Silicon (MPS) generation can differ between runs even with the same seed.
- `configs/ppo_debate_judge.yaml` audits with `Qwen/Qwen2.5-3B-Instruct`, because the debate arbiter and the default gold judge are both `mistralai/Mistral-7B-Instruct-v0.2`. Gold agreement for debate runs therefore uses a different gold judge from the other architectures and is not directly comparable with theirs. Qwen2.5-3B shares a family with the Qwen2.5-1.5B judge and advocate, though not with the arbiter.
- The debate pair has no in-training evaluation round, only the gold audit. `build_results_tables.py` labels a debate run's `judge_architecture` as `single`, because it only looks at the ensemble and escalation fields. Its `algo` is `ppo_multiagent`, so it still forms its own group.
- `mean_steps` for the debate pair counts the 10 role turns and not the forced `FINALIZE`, while the single-agent evaluator counts `FINALIZE` as a step.
