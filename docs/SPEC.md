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

- Thirteen actions (`environment/actions.py`): `SELECT`, `REMOVE`, `SUPPORT`, `CONTRADICT`, `FINALIZE`, `QUERY`, `RERANK`, `SUMMARIZE`, `CONCEDE`, `ASSIGN_CONFIDENCE`, `CHALLENGE_EVIDENCE`, `REQUEST_CLARIFICATION`, `HEDGE`. The README lists nine. **[inferred]**
- `QUERY` and `REQUEST_CLARIFICATION` share a budget of `max_queries` per episode (README says 2). **[inferred]**
- The judge is called at most once every two steps. In between, the previous score is reused. **[inferred]**
- Episodes end on `FINALIZE` with at least one selected evidence item, or at the step limit. The README says 10 and the limit lives in `State.is_done`. **[inferred]**
- `FINALIZE` with no evidence ends the episode at -1.0. `FINALIZE` at step 2 or earlier gives -0.5 and the episode continues. **[inferred]**
- Every step reward gets `0.1 * (0.99 * phi_next - phi_prev)`, where phi is the last judge score and is 0 at terminal states. **[inferred]**
- `task_success = 1 - |confidence - label|` is set only on `FINALIZE`. The curriculum and gold eval use it. **[inferred]**
- `confidence` is the agent's `ASSIGN_CONFIDENCE` value, or `min(1, n_selected / 3)` if it never called it. **[inferred]**

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
- Results are reported through the Streamlit dashboard over `artifacts/experiments/`. A committed results table is not set up yet. **[confirmed, open question]**

## 8. Known gaps between README and code

- The README says nine actions. The code has thirteen.
- The README's per-action step reward table does not match `ClaimEnv.step`. `SELECT` is `_SELECT_LABEL_REWARD` plus a diversity bonus with no claim-similarity term, and `QUERY` is `min(0.15, 0.05*useful)`.
- The README says `base_reward` is clipped to [0, 1] and puts the final guards before a `0.7/0.3` blend. The code clips to [-1, 1] and has no `0.7/0.3` blend.
- The seed-42 split lives in `data/dataset.py` (`split_dataset`) and is used by train, eval and collect. The collect copy failed to restore global RNG state until commit 1d98db8, which made `evid-collect --seed` ineffective before then.
- `ClaimEnv.reset` samples with the global `random.choice`, so reproducibility depends on the caller's seeding.
