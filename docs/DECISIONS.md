# Decisions

One entry per decision that someone might later question. Newest last.
Entries are drafts from the code. Where the reason is a guess, the text says "probably".

## D-001: Claims carry gold evidence and Tavily is the fallback

SciFact claims carry human-labeled abstracts, so the environment uses them directly and skips Tavily and the LLM labeler. Probably done to make evidence labels reliable and runs reproducible. Live retrieval stays for claims without gold evidence.

## D-002: One seed-42 80/20 split, shared by train, eval and collect

The split is a shuffled index cut at 80%, with global RNG state saved and restored so it does not disturb training seeds. A leakage bug (commit "fixed data leakage") led to seeding it separately. The logic now lives in one helper, `split_dataset` in `data/dataset.py`. Before that it was copied into three CLIs, and the `evid-collect` copy only started restoring RNG state in commit 1d98db8.

## D-003: Gold judges are separate from the training judge

`GoldEvaluator` re-scores final reasoning with a different, larger model (Mistral-7B) that never feeds training reward. It exists to audit the training judge as a proxy.

## D-004: Gold eval samples stochastically

Greedy rollouts from an under-trained policy can get stuck and never call `FINALIZE`, which would leave nothing to score. So gold eval uses `greedy=False`, unlike the standard evaluator.

## D-005: The judge runs at most every second step

This limits inference cost. Between calls the last score is reused, so the delta is zero on skipped steps.

## D-006: Potential-based shaping with the judge score as potential

`0.1 * (0.99 * phi_next - phi_prev)`, with phi set to 0 at terminal states so the shaping telescopes. It follows Ng et al. 1999.

## D-007: Final reward is mostly heuristic, with a small judge weight

The judge weight is 0.10 in `RewardFunction` and there is no further blend. The README still describes a 0.7/0.3 blend, which is stale.

## D-008: Judge scores blend toward neutral by confidence

`conf * reward + (1 - conf) * 0.5` replaces a hard threshold, because the threshold made the reward discontinuous. An all-0.5 score is treated as a parse-failure sentinel and returns 0.5.

## D-009: Template-echo and parse failures return neutral scores

Small judge models sometimes echo the prompt template with all zeros. That is detected and replaced with neutral scores. Transient generation failures are not cached.

## D-010: Judge architectures are composable

Ensemble, debate and escalating judges wrap the single judge and are selected by config, not by code branches in the trainers. Escalation sends hard cases from a cheap judge to a costlier tier.

## D-011: Per-claim Prioritized Level Replay for the curriculum

The curriculum scores each claim by learning progress, staleness and a floor weight, using `task_success` rather than value loss, so it works across bandit, PG and PPO.

## D-012: Hyperparameters live in YAML, not in Python config classes

The config classes hold only the schema and cross-cutting defaults, so tuning means editing a YAML file.

## D-013: SQLite caches for judge, label and fetch results

Chosen over POSIX-semaphore-based caches after a semaphore leak (commit "fixed semaphore leak"). Caches are gitignored, so results are stable on one machine but not portable. `evid-snapshot` is the portable route.

## D-014: Tests mock every model boundary

`ClaimEnv` takes injected `llm_judge`, `embedder` and `evidence_labeler`, so the suite runs with no network, keys or model downloads.

## D-015: ClaimEnv samples claims from its own RNG

`ClaimEnv.reset` used the global `random.choice`, and the baselines drew their fallback actions from the same stream. The claim a method saw therefore depended on how many draws earlier methods had made, so two methods with the same seed could be evaluated on different episodes. Each env now owns a `random.Random(seed)`, evaluators reseed it every round, and the baselines are built with the evaluation seed. The curriculum got its own RNG for the same reason, `train.py` seeds numpy before the policy draws its initial weights (this used to happen only as a side effect of building the judge client), and baselines are reseeded before each run. The cost is that runs made before this change cannot be replayed exactly.

## D-016: The debate config audits with a different gold judge

The debate judge's arbiter is Mistral-7B-Instruct-v0.2, which is also the default gold judge, so a debate run's gold audit would score its reasoning with the model that produced its reward. `ppo_debate_judge.yaml` now sets `gold_judge_model` to Qwen2.5-3B-Instruct. A test checks that no config's gold judge is one of its training judges. The other local candidate, prometheus-7b-v2.0, is probably built on the same Mistral base, so I did not use it. The cost is that debate gold agreement is not directly comparable with the other architectures. `evid-gold-eval --gold-judge-model` can rescore a checkpoint with a second gold judge if that comparison is needed.

## D-017: Rescoring replays recorded actions with rebuilt payloads

`evid-collect --mode rescore` re-runs the actions of an existing trajectory file under the current environment and reward. Trajectories store the action index but not the payload, for example which document a `SELECT` picked, so the payloads are rebuilt with the global RNG seeded from `--seed`, the record's position and its claim, and the RNG state is restored afterwards. The new reward is the reward of that seeded replay, not of the original episode, and the old reward is kept as `source_reward`. Records for claims outside the training split are skipped, so an old file cannot carry eval claims into the imitation data. The output is a new file with `mode: rescored`, and the command refuses to overwrite its input. `--relabel` adds the annotator's action at each step as `annotator_action` and never changes the action that was executed.

## D-018: Multi-agent debate is a separate environment wrapper, trainer and evaluator

Two policies argue one claim, a support role and a contradict role, and an arbiter scores the transcript. `ClaimEnv` is untouched, so existing runs and their results do not move. `DebateEnv` wraps it, owns turn order, and runs `FINALIZE` itself after a fixed 5 turns per side. The arbiter goes in as the env's `llm_judge`, so the same reward function applies and only the judge term changes. The roles share the evidence pool, the selected set and the transcript, so `State` is unchanged. The arbiter's final reward is added to each role's last step and the roles are updated separately with PPO. The alternative was a zero-sum split, which rewards winning but can push both sides toward overclaiming.

Role limits are action masks: the support role cannot use `CONTRADICT`, the contradict role cannot use `SUPPORT`, and neither can use `FINALIZE`. The last turn must assign a confidence if none was set, so the final score does not fall back to the evidence-count guess. The fallback stays as a backstop. The step limit is raised to 11 for debate episodes, because the limit would otherwise replace the arbiter's score with -0.2.

The arbiter model is set by `arbiter_model` in `configs/ppo_multiagent_debate.yaml` and is Qwen2.5-3B-Instruct, so the default Mistral-7B gold judge stays independent. It was SmolLM2-1.7B-Instruct first. See D-020. The gold-judge guard test now also reads `arbiter_model`. Both policies share one loaded actor model and differ only in their weights. `DebateTrainer` has no in-training evaluation round, because the existing evaluators play one policy. `DebateEvaluator` plays the pair on the shared single-judge eval env with the evid-eval seed, so the pair is comparable with the single-agent runs.

The gold audit reuses `GoldEvaluator` unchanged by presenting the pair to it as one env and one policy (`agent/debate_gold.py`). The proxy side is the arbiter's final score and the gold judge re-scores the same two-sided transcript. The trainer runs it every `eval_every * gold_eval_every` episodes, and `python -m evid_rl_env.cli.eval_debate --gold-judge-model ...` runs it on a saved run. Both refuse a gold judge that is the arbiter.

A debate has a token budget, `max_tokens_per_episode` in the config. A debate that reaches it stops and is still finalized, so the arbiter always scores it and the episode is flagged `truncated`. The single-agent rule of ending the episode with -0.2 would have skipped the arbiter. Argument text is capped at 128 new tokens per turn and the count is in words, so ten turns cannot approach 2000 and the cap is a safety net, set to 6000.

## D-019: Updates use the features and mask stored at act time, and SUPPORT writes only its argument

Three bugs were fixed together because they all changed what a PPO or PG run learns from. `ClaimEnv` mutates one `State` in place, so every trajectory entry pointed at the same object and the updaters re-encoded the end-of-episode state for every step. The policy now stores its features and action mask in `act`, the trainer keeps them per step, and both updaters take them instead of states. `act` also masked some actions after the softmax while the updaters used a plain softmax, so the PPO ratio and entropy came from a different distribution than the one sampled. One `masked_softmax` now serves `act`, `PPO.update` and `PolicyGradient.update`. Finally, `SUPPORT` and `CONTRADICT` appended `str(payload)` to the transcript, so the judge read a dict repr with evidence ids and a token count. They now append the argument text, like the other argument actions. Every run made before these changes is invalid, bandit runs included, because the transcript text changed the judge-scored reward.

## D-020: The debate arbiter is Qwen2.5-3B-Instruct

The first arbiter was SmolLM2-1.7B-Instruct. In the smoke run its scores were often the neutral 0.5 that `LLMJudge` substitutes when the model echoes the score template. I compared it with Qwen2.5-3B-Instruct on one set of debates. Gemma-2-2b wrote four arguments, alternating support and contradict, for each of 12 claims from the train split, and each candidate scored every debate once with its own claim and once with the arguments of a different claim. SmolLM2 gave 17 clean JSON outputs out of 24, 5 neutral fallbacks, and scored the matched debate higher in 5 of 12 pairs with 1 tie. Qwen2.5-3B gave 24 clean outputs, no fallbacks, and scored the matched debate higher in 9 of 12 pairs. The mean reward was 0.499 for matched and 0.429 for mismatched debates. Twelve pairs is a small sample, so the format result is firm and the discrimination result is suggestive. The check ran from a one-off script outside the repo and its output is not committed. Mistral-7B stays the gold judge, so the arbiter and gold judge differ. A mixture-of-experts arbiter was not tried, because it needs the same memory as its total parameter count and runs less reliably through `transformers` on MPS.

