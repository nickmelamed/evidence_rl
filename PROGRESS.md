# Progress

Current phase and what is left. Claude updates this at the end of each
phase. CLAUDE.md stays stable and does not track status. The "Now" section
is shown to Claude at the start of each session and after compaction.

## Now

- [ ] Review and merge `feat/llm-call-seeding` (per-call seeding, environment record, gold eval raised to 100, docs). Not pushed yet.
- [ ] Decide the debate-judge overlap below before training the debate architecture.
- [ ] Budget the reruns (TODO below), then start them from HANDOFF.md.
- [ ] Edit docs/SPEC.md and docs/DECISIONS.md into your own words. Entries marked inferred in SPEC are guesses from the code.

## Rerun plan

The full instructions for a fresh conversation are in HANDOFF.md. In short: train one PPO run per judge architecture with seed 42, evaluate each checkpoint against the baselines, then run `make results` and update the README Results section.

State as of 2026-10-06:

- `gold_eval_n_episodes` is 100 in `configs/base.yaml` (was 20). Most old gold rounds scored 0 to 5 episodes, because few stochastic rollouts reach `FINALIZE`. More episodes helps, but the finalize rate is the real limit.
- The judge caches are emptied. The old files are in `artifacts/cache_archive/judge_caches_20261006/` (gitignored). The evidence and fetch caches are untouched, since claims carry gold evidence.
- Old rows stay in the results tables, since they are built from every folder in `artifacts/experiments/`. Move the old folders aside first if the tables should show only the new runs.
- Claim order is fixed by the seed, so runs made before `feat/env-rng` cannot be compared with the new ones.

Budget TODO: estimate wall-clock time and disk for the reruns before launching. Inputs we have: a 10-episode PPO run with one eval round took 13m47s with a cold judge cache and 3m42s with a warm one. Earlier 300-episode runs took roughly a day each (from folder timestamps, not measured). Gold eval at 100 episodes with a 7B judge every 5 eval rounds is now a large share of each run. Decide the number of architectures, seeds and episodes from that estimate.

Blocking decision: the debate judge's arbiter is `mistralai/Mistral-7B-Instruct-v0.2` (hard-coded in `judge/debate_judge.py`), which is also the default `gold_judge_model`. For `ppo_debate_judge.yaml` and any escalating-to-debate run, the gold judge is the same model that produced the training reward, so proxy-vs-gold agreement is not independent. This breaks rule 2 in CLAUDE.md for those runs. Options: use a different gold judge (the cache has `prometheus-eval/prometheus-7b-v2.0` and `Qwen/Qwen2.5-3B-Instruct`), change the arbiter, or leave the debate architecture out of the gold comparison. Nothing has been changed.

## Determinism on MPS

Done: torch is seeded before every generation from the run seed, the prompt and how often that prompt was seen (`derive_call_seed` in `agent/llm_client.py`), and `config.json` records the device and library versions. A sample no longer depends on call order or on judge cache hits.

Smoke test (2026-10-06, MPS, torch 2.11.0, transformers 5.5.4): two 10-episode PPO runs with seed 42, the second with a warmer judge cache (13m47s against 3m42s). Every per-episode reward, step count, token count and entropy matched, the eval table matched, and the saved policy weights were identical. Ten episodes on one machine is a small test, so longer runs could still drift.

Not done, to revisit if the smoke test or the reruns show the variation matters:

- [ ] Greedy decoding for the judge (`do_sample=False`). It makes scoring deterministic but slightly changes the reward distribution, so it needs a decision.
- [ ] Several seeds per configuration (3 to 5), reported as mean and spread. This matters more than bitwise determinism, and it costs compute.
- [ ] Final runs on one deterministic device. CPU with float32 is slow for 2B to 7B models. CUDA with `torch.use_deterministic_algorithms(True)` is deterministic but gives different numbers from MPS.
- [ ] Pin the model dtype instead of `torch_dtype="auto"`, since half precision adds numeric noise.

## Next


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

## Open questions for the owner

- Trajectory and log files made by `evid-collect` before `1d98db8` replayed the same random stream whatever `--seed` was. They were not regenerated. Decide whether any imitation data or results that depended on them need recollecting.
- Runs made before `feat/env-rng` cannot be replayed exactly, because their claim order came from the global RNG.
- mypy runs with `ignore_missing_imports = true`, because several dependencies ship no stubs.
