# Decisions

One entry per decision that someone might later question. Newest last.
These entries are drafts written from the code. Edit them into your own words.
Where the reason is a guess, the text says "probably".

## D-001: Claims carry gold evidence and Tavily is the fallback

SciFact claims carry human-labeled abstracts, so the environment uses them directly and skips Tavily and the LLM labeler. Probably done to make evidence labels reliable and runs reproducible. Live retrieval stays for claims without gold evidence.

## D-002: One seed-42 80/20 split, shared by train, eval and collect

The split is a shuffled index cut at 80%, with global RNG state saved and restored so it does not disturb training seeds. The fix is commented as an audit fix after a leakage bug (commit "fixed data leakage"). It is copied into three files rather than shared.

## D-003: Gold judges are separate from the training judge

`GoldEvaluator` re-scores final reasoning with a different, larger model (Mistral-7B) that never feeds training reward. It exists to audit the training judge as a proxy.

## D-004: Gold eval samples stochastically

Greedy rollouts from an under-trained policy can get stuck and never call `FINALIZE`, which would leave nothing to score. So gold eval uses `greedy=False`, unlike the standard evaluator.

## D-005: The judge runs at most every second step

This limits inference cost. Between calls the last score is reused, so the delta is zero on skipped steps.

## D-006: Potential-based shaping with the judge score as potential

`0.1 * (0.99 * phi_next - phi_prev)`, with phi set to 0 at terminal states so the shaping telescopes. It follows Ng et al. 1999 and was added after the "reward hacking" fix.

## D-007: Final reward is mostly heuristic, with a small judge weight

The judge weight is 0.10 in `RewardFunction`, and the older extra re-blend was removed so the documented weight is the real one. The README still describes a 0.7/0.3 blend. This is a known mismatch.

## D-008: Judge scores blend toward neutral by confidence

`conf * reward + (1 - conf) * 0.5` replaces a hard threshold, because the threshold made the reward discontinuous. An all-0.5 score is treated as a parse-failure sentinel and returns 0.5.

## D-009: Template-echo and parse failures return neutral scores

Small judge models sometimes echo the prompt template with all zeros. That is detected and replaced with neutral scores. Transient generation failures are not cached.

## D-010: Judge architectures are composable

Ensemble, debate and escalating judges wrap the single judge and are selected by config, not by code branches in the trainers. Escalation sends hard cases from a cheap judge to a costlier tier.

## D-011: Per-claim Prioritized Level Replay for the curriculum

It scores each claim by learning progress, staleness and a floor weight, using `task_success` rather than value loss, so it works across bandit, PG and PPO.

## D-012: Hyperparameters live in YAML, not in Python config classes

The config classes hold only the schema and cross-cutting defaults, so tuning means editing a YAML file.

## D-013: SQLite caches for judge, label and fetch results

Chosen over POSIX-semaphore-based caches after a semaphore leak (commit "fixed semaphore leak"). Caches are gitignored, so results are stable on one machine but not portable. `evid-snapshot` is the portable route.

## D-014: Tests mock every model boundary

`ClaimEnv` takes injected `llm_judge`, `embedder` and `evidence_labeler`, so the suite runs with no network, keys or model downloads.
