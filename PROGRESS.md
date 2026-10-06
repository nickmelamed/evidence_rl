# Progress

Current phase and what is left. Claude updates this at the end of each
phase. CLAUDE.md stays stable and does not track status. The "Now" section
is shown to Claude at the start of each session and after compaction.

## Now

- [ ] Review and merge `chore/open-items` (not pushed yet).
- [ ] Edit docs/SPEC.md and docs/DECISIONS.md into your own words. Entries marked inferred in SPEC are guesses from the code.

## Next

- [ ] Rerun the PPO, PG and bandit evaluations. The three runs in `results/tables/eval_baselines.csv` predate the reward and judge changes of July 2026.
- [ ] Raise `gold_eval_n_episodes`, since most logged gold rounds scored 0 to 5 episodes and say little about judge agreement.
- [ ] Turn on mypy's `check_untyped_defs` and work down what it finds. The current check skips the bodies of untyped functions.
- [ ] Enable the project-checks step in `.github/workflows/agent-checks.yml` once `make ci` should run in CI.
- [ ] Add `ruff format --check` only if you want the repo reformatted.

## Done

- [x] Phase 0 to 5: assessment, spec, standards installed, style cleanup, safety net, merged as PR #1.
- [x] Fixed `evid-collect --seed` being discarded (`1d98db8`).
- [x] One `split_dataset` helper in `data/dataset.py` replaces three copies, pinned to the old split by a hash test.
- [x] Duplicate claim text (`scifact_85`, `scifact_86`) kept, with a test that it never straddles the split.
- [x] `results/tables/` built by `make results`, README quotes only those numbers, `make numbers` checks it and runs in `make ci`.
- [x] README matches the code (13 actions, real step and final rewards).
- [x] mypy passes and is in `make agent-check` and `make ci`.

## Open questions for the owner

- Trajectory and log files made by `evid-collect` before `1d98db8` replayed the same random stream whatever `--seed` was. They were not regenerated. Decide whether any imitation data or results that depended on them need recollecting.
- `ClaimEnv.reset` still samples with the global `random.choice`. Giving it its own RNG would change which episodes each seed produces, so past runs would no longer match exactly.
- mypy runs with `ignore_missing_imports = true`, because several dependencies ship no stubs.
