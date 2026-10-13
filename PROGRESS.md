# Progress

Current phase and what is left. Claude updates this at the end of each
phase. CLAUDE.md stays stable and does not track status. The "Now" section
is shown to Claude at the start of each session and after compaction.

## Now

- [ ] Multi-agent debate (Future Work item 2) is built and tested with mocks on `feat/multi-agent-debate`. It has not been run with real models. A tiny end-to-end run needs your approval, and the PR needs review.
- [ ] Run the reruns from HANDOFF.md, Task 2, in a separate conversation. The update fixes are on `main`, so they can start.
- [ ] Edit docs/SPEC.md and docs/DECISIONS.md into your own words. Entries marked inferred in SPEC are guesses from the code.

## Future Work status

The README listed four Future Work items. Item 2 is removed from it on the debate branch. Their state as of 2026-10-12:

1. Re-annotation pipeline: done. `evid-collect --mode rescore` replays recorded actions under the current environment and reward, and `--relabel` adds a strong LLM's action per step (D-017). Checked end to end on 3 trajectories against the real environment and the real Anthropic API.
2. Multi-agent debate: built and mock-tested on `feat/multi-agent-debate` (D-018), not yet run with real models. See HANDOFF.md, Task 1.
3. Learned reward models: agreed scope is scaffolding only. A pluggable reward model judge type plus training code, tested on synthetic preference pairs. A trained model needs human preference data that this repo does not have.
4. Domain expansion: agreed scope is scaffolding only. A domain registry, a retriever interface and a per-domain split contract. It needs labeled datasets for each domain, which this repo does not have.

Items 3 and 4 change the reward and the data loading, so each needs a short plan and your approval before code.

## Rerun plan

The full instructions for a fresh conversation are in HANDOFF.md. In short: train four PPO judge architectures (baseline, ensemble, escalation, debate) with three seeds each (42, 43, 44), evaluate every checkpoint on the same held-out episodes, then run `make results` and update the README Results section from `results/tables/`.

State as of 2026-10-06:

- `gold_eval_n_episodes` is 100 in `configs/base.yaml` (was 20). Most old gold rounds scored 0 to 5 episodes, because few stochastic rollouts reach `FINALIZE`. More episodes helps, but the finalize rate is the real limit.
- The judge caches are emptied. The old files are in `artifacts/cache_archive/judge_caches_20261006/` (gitignored). The evidence and fetch caches are untouched, since claims carry gold evidence.
- The debate config audits with `Qwen/Qwen2.5-3B-Instruct` instead of the default gold judge, because the debate arbiter is the default gold judge (D-016). A test checks that no config's gold judge is one of its training judges. Debate gold agreement is not directly comparable with the other architectures.
- Tavily has auto-reload on, so running out of credits is not a concern.
- Three seeds per architecture, as decided. The order and the number of seeds can be adjusted once the first runs show how long they take.
- `results/tables/summary_by_architecture.csv` groups runs by algorithm and judge architecture and reports the mean and sample standard deviation across seeds.
- The 13 old experiment folders were moved to `artifacts/experiments_archive_pre_seeding/` on 2026-10-13 with the owner's approval, so `make results` will only see new runs. The committed tables in `results/tables/` still describe the old runs until the reruns rebuild them.
- Claim order is fixed by the seed, so runs made before `feat/env-rng` cannot be compared with the new ones.
- Every PPO and PG run made before the two fixes under Done (2026-10-13) is invalid, not only the pre-seeding ones. Bandit runs were not affected. The fixes are merged to `main`. A third fix (2026-10-13) changes judge-scored rewards for every algorithm, bandit included: `SUPPORT` and `CONTRADICT` used to put the whole payload dict, with evidence ids and the token count, into the transcript the judge reads, and now put in only the argument text. It is merged, so the reruns can start.

Budget TODO: record the wall-clock time and disk use of the first finished run here, and extrapolate to the rest. Inputs we have: a 10-episode PPO run with one eval round took 13m47s with a cold judge cache and 3m42s with a warm one. Earlier 300-episode runs took roughly a day each (from folder timestamps, not measured). Gold eval at 100 episodes with a 7B judge every 5 eval rounds is a large share of each run. Twelve runs at that rate is a long job, so cut seeds or episodes if the first run confirms it.

## Determinism on MPS

Done: torch is seeded before every generation from the run seed, the prompt and how often that prompt was seen (`derive_call_seed` in `agent/llm_client.py`), and `config.json` records the device and library versions. A sample no longer depends on call order or on judge cache hits.

Smoke test (2026-10-06, MPS, torch 2.11.0, transformers 5.5.4): two 10-episode PPO runs with seed 42, the second with a warmer judge cache (13m47s against 3m42s). Every per-episode reward, step count, token count and entropy matched, the eval table matched, and the saved policy weights were identical. Ten episodes on one machine is a small test, so longer runs could still drift.

Not done, to revisit if the smoke test or the reruns show the variation matters:

- [ ] Greedy decoding for the judge (`do_sample=False`). It makes scoring deterministic but slightly changes the reward distribution, so it needs a decision.
- [ ] Final runs on one deterministic device. CPU with float32 is slow for 2B to 7B models. CUDA with `torch.use_deterministic_algorithms(True)` is deterministic but gives different numbers from MPS.
- [ ] Pin the model dtype instead of `torch_dtype="auto"`, since half precision adds numeric noise.

## Next

- [ ] Scaffolding for Future Work items 3 and 4, after item 2.
- [ ] Rerun the PPO, PG and bandit evaluations. The three runs in `results/tables/eval_baselines.csv` predate the reward and judge changes of July 2026.
- [ ] Add `ruff format --check` only if you want the repo reformatted.

## Done

- [x] Phase 0 to 5: assessment, spec, standards installed, style cleanup, safety net, merged as PR #1.
- [x] Fixed `evid-collect --seed` being discarded (`1d98db8`).
- [x] One `split_dataset` helper in `data/dataset.py` replaces three copies, pinned to the old split by a hash test.
- [x] Duplicate claim text (`scifact_85`, `scifact_86`) kept, with a test that it never straddles the split.
- [x] `results/tables/` built by `make results`, README quotes only those numbers, `make numbers` checks it and runs in `make ci`.
- [x] README matches the code (13 actions, real step and final rewards).
- [x] mypy passes with `check_untyped_defs` and is in `make agent-check` and `make ci`.
- [x] CI runs `make install-dev && make ci` in `agent-checks.yml`.
- [x] Per-call torch seeding, environment record in `config.json`, determinism smoke test.
- [x] Judge caches emptied and archived, gold eval raised to 100 episodes.
- [x] Debate config uses a different gold judge (D-016), with a guard test.
- [x] Results tables report mean and spread across seeds (`summary_by_architecture.csv`).
- [x] `evid-collect --mode rescore` and `--relabel` (D-017). `--relabel` exits without `ANTHROPIC_API_KEY`, and relabeling never changes the replayed rewards.
- [x] `AnnotatorClient.select_action` raises `AnnotatorError` on an API or parse failure. It used to return a random action that `evid-collect` stored as the model's label.
- [x] Multi-agent debate code (2026-10-13): role masks, `Arbiter`, `DebateEnv`, `DebateTrainer`, `ppo_multiagent_debate.yaml`, `DebateEvaluator` and `eval_debate`. Mock-tested only.
- [x] `SUPPORT` and `CONTRADICT` write only the argument text to `debate_history` (2026-10-13). They used `str(payload)`, so the judge and the gold judge read a dict repr. Every earlier run, bandit included, was scored on that text.
- [x] Fixed the PPO and PG updates (2026-10-13). `ClaimEnv` mutates one `State` in place, so every trajectory entry held the end-of-episode state and each step was updated with the wrong inputs (`7426705`). The updaters also used a plain softmax while `act` sampled from a masked one, so the PPO ratio and entropy were computed on the wrong distribution (`cce5354`). Features and the action mask are now stored at act time and shared through `masked_softmax`.

## Open questions for the owner

- Trajectory and log files made by `evid-collect` before `1d98db8` replayed the same random stream whatever `--seed` was. They were not regenerated. Decide whether any imitation data or results that depended on them need recollecting.
- Runs made before `feat/env-rng` cannot be replayed exactly, because their claim order came from the global RNG.
- Imitation files made by `evid-collect --mode llm_annotator` before the `AnnotatorError` change may contain random actions labeled as the annotator's, wherever an API call failed. The old client only logged a warning. Check the logs for "returning random" before using them.
- mypy runs with `ignore_missing_imports = true`, because several dependencies ship no stubs.
