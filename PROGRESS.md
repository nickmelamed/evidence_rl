# Progress

Current phase and what is left. Claude updates this at the end of each
phase. CLAUDE.md stays stable and does not track status. The "Now" section
is shown to Claude at the start of each session and after compaction.

## Now

- [ ] Run the reruns from HANDOFF.md in a separate conversation.
- [ ] Edit docs/SPEC.md and docs/DECISIONS.md into your own words. Entries marked inferred in SPEC are guesses from the code.

## Rerun plan

The full instructions for a fresh conversation are in HANDOFF.md. In short: train four PPO judge architectures (baseline, ensemble, escalation, debate) with three seeds each (42, 43, 44), evaluate every checkpoint on the same held-out episodes, then run `make results` and update the README Results section from `results/tables/`.

State as of 2026-10-06:

- `gold_eval_n_episodes` is 100 in `configs/base.yaml` (was 20). Most old gold rounds scored 0 to 5 episodes, because few stochastic rollouts reach `FINALIZE`. More episodes helps, but the finalize rate is the real limit.
- The judge caches are emptied. The old files are in `artifacts/cache_archive/judge_caches_20261006/` (gitignored). The evidence and fetch caches are untouched, since claims carry gold evidence.
- The debate config audits with `Qwen/Qwen2.5-3B-Instruct` instead of the default gold judge, because the debate arbiter is the default gold judge (D-016). A test checks that no config's gold judge is one of its training judges. Debate gold agreement is not directly comparable with the other architectures.
- Tavily has auto-reload on, so running out of credits is not a concern.
- Three seeds per architecture, as decided. The order and the number of seeds can be adjusted once the first runs show how long they take.
- `results/tables/summary_by_architecture.csv` groups runs by algorithm and judge architecture and reports the mean and sample standard deviation across seeds.
- Old rows stay in the results tables, since they are built from every folder in `artifacts/experiments/`. Move the old folders aside before the first new run, with the owner's approval.
- Claim order is fixed by the seed, so runs made before `feat/env-rng` cannot be compared with the new ones.

Budget TODO: record the wall-clock time and disk use of the first finished run here, and extrapolate to the rest. Inputs we have: a 10-episode PPO run with one eval round took 13m47s with a cold judge cache and 3m42s with a warm one. Earlier 300-episode runs took roughly a day each (from folder timestamps, not measured). Gold eval at 100 episodes with a 7B judge every 5 eval rounds is a large share of each run. Twelve runs at that rate is a long job, so cut seeds or episodes if the first run confirms it.

## Determinism on MPS

Done: torch is seeded before every generation from the run seed, the prompt and how often that prompt was seen (`derive_call_seed` in `agent/llm_client.py`), and `config.json` records the device and library versions. A sample no longer depends on call order or on judge cache hits.

Smoke test (2026-10-06, MPS, torch 2.11.0, transformers 5.5.4): two 10-episode PPO runs with seed 42, the second with a warmer judge cache (13m47s against 3m42s). Every per-episode reward, step count, token count and entropy matched, the eval table matched, and the saved policy weights were identical. Ten episodes on one machine is a small test, so longer runs could still drift.

Not done, to revisit if the smoke test or the reruns show the variation matters:

- [ ] Greedy decoding for the judge (`do_sample=False`). It makes scoring deterministic but slightly changes the reward distribution, so it needs a decision.
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
- [x] Debate config uses a different gold judge (D-016), with a guard test.
- [x] Results tables report mean and spread across seeds (`summary_by_architecture.csv`).

## Open questions for the owner

- Trajectory and log files made by `evid-collect` before `1d98db8` replayed the same random stream whatever `--seed` was. They were not regenerated. Decide whether any imitation data or results that depended on them need recollecting.
- Runs made before `feat/env-rng` cannot be replayed exactly, because their claim order came from the global RNG.
- mypy runs with `ignore_missing_imports = true`, because several dependencies ship no stubs.
