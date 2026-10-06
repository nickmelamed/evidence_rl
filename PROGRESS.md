# Progress

Current phase and what is left. Claude updates this at the end of each
phase. CLAUDE.md stays stable and does not track status. The "Now" section
is shown to Claude at the start of each session and after compaction.

## Now

- [ ] Review the `feat/env-rng` change (not pushed yet).
- [ ] Edit docs/SPEC.md and docs/DECISIONS.md into your own words. Entries marked inferred in SPEC are guesses from the code.

## Rerun plan

Run these from the repo root inside the `ev_rl` venv, with `TAVILY_API_KEY` set in `.env`. The seed is 42 everywhere, from `configs/base.yaml`.

1. Train one run per judge architecture, with the same episode count and seed:
   `evid-train --config configs/ppo_baseline.yaml --episodes 300 --exp-name full_baseline`
   `evid-train --config configs/ppo_ensemble_judge.yaml --episodes 300 --exp-name full_ensemble`
   `evid-train --config configs/ppo_escalation_judge.yaml --episodes 300 --exp-name full_escalation`
   `evid-train --config configs/ppo_debate_judge.yaml --episodes 300 --exp-name full_debate`
   Add `evid-train --config configs/pg_baseline.yaml` and `configs/bandit_baseline.yaml` if you want the three algorithms again.
2. Evaluate each checkpoint against the baselines. The eval env uses the default single judge, so every architecture is scored on the same yardstick:
   `evid-eval --checkpoint artifacts/experiments/<run>/policy.npz --baselines random,majority,greedy_llm,fewshot_k3,best_of_5 --n-episodes 100`
3. Raise `gold_eval_n_episodes` in `configs/base.yaml` before training if you want usable gold-judge numbers. Most past rounds scored 0 to 5 episodes. That file is a protected path.
4. Run `make results`, then update the README Results section from `results/tables/`. `make numbers` fails until the quoted numbers match the tables.
5. Old rows stay in the tables, since they are built from every folder in `artifacts/experiments/`. Move or delete the old folders first if you want the tables to show only the new runs.

Claim order is now fixed by the seed, so two methods evaluated with the same seed see the same claims. Do not compare these runs with ones made before `feat/env-rng`.

## Determinism on MPS

Done: torch is seeded before every generation from the run seed, the prompt and how often that prompt was seen (`derive_call_seed` in `agent/llm_client.py`), and `config.json` records the device and library versions. A sample no longer depends on call order or on judge cache hits.

Smoke test (2026-10-06, MPS, torch 2.11.0, transformers 5.5.4): two 10-episode PPO runs with seed 42, the second with a warmer judge cache (13m47s against 3m42s). Every per-episode reward, step count, token count and entropy matched, the eval table matched, and the saved policy weights were identical. Ten episodes on one machine is a small test, so longer runs could still drift.

Not done, to revisit if the smoke test or the reruns show the variation matters:

- [ ] Greedy decoding for the judge (`do_sample=False`). It makes scoring deterministic but slightly changes the reward distribution, so it needs a decision.
- [ ] Several seeds per configuration (3 to 5), reported as mean and spread. This matters more than bitwise determinism, and it costs compute.
- [ ] Final runs on one deterministic device. CPU with float32 is slow for 2B to 7B models. CUDA with `torch.use_deterministic_algorithms(True)` is deterministic but gives different numbers from MPS.
- [ ] Pin the model dtype instead of `torch_dtype="auto"`, since half precision adds numeric noise.
- [ ] Empty the judge cache before the final runs so they reproduce from cold. `artifacts/cache/judge_cache*.sqlite3` persists across runs.

## Next


- [ ] Rerun the PPO, PG and bandit evaluations. The three runs in `results/tables/eval_baselines.csv` predate the reward and judge changes of July 2026.
- [ ] Raise `gold_eval_n_episodes`, since most logged gold rounds scored 0 to 5 episodes and say little about judge agreement.
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

## Open questions for the owner

- Trajectory and log files made by `evid-collect` before `1d98db8` replayed the same random stream whatever `--seed` was. They were not regenerated. Decide whether any imitation data or results that depended on them need recollecting.
- Runs made before `feat/env-rng` cannot be replayed exactly, because their claim order came from the global RNG.
- mypy runs with `ignore_missing_imports = true`, because several dependencies ship no stubs.
