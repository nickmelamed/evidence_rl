# Progress

Current phase and what is left. Claude updates this at the end of each
phase. CLAUDE.md stays stable and does not track status. The "Now" section
is shown to Claude at the start of each session and after compaction.

## Now

- [ ] Review and merge `chore/agent-standards` (nothing pushed yet).
- [ ] Edit docs/SPEC.md and docs/DECISIONS.md into your own words. Entries marked inferred in SPEC are guesses from the code.

## Next

- [ ] Bring the README in line with the code: 13 actions, the real step rewards, the real final reward (clip to [-1, 1], no 0.7/0.3 blend). See SPEC section 8.
- [ ] Decide how results are reported. Proposal: `evid-eval` writes a table under `results/tables/`, the README quotes only that table, and `make numbers` runs `check_numbers.py`. The `results-table` skill does the build.
- [ ] Replace the invented example eval table in the README with real output or label it illustrative.
- [ ] Consolidate the three copies of the seed-42 split (`cli/train.py`, `cli/eval.py`, `cli/collect_trajectories.py`) into one helper.
- [ ] Make `ClaimEnv.reset` use a seeded RNG instead of the global `random.choice`.
- [ ] Work down the 140 mypy errors, then add `make typecheck` to `agent-check`.
- [ ] Decide what to do about the duplicated claim text (`scifact_85` and `scifact_86`) before it can straddle the split.
- [ ] Add `ruff format --check` only if you want the repo reformatted.
- [ ] Enable the project-checks step in `.github/workflows/agent-checks.yml` once `make ci` is wanted in CI.

## Done

- [x] Phase 0: assessment. Retrofit chosen.
- [x] Phase 1: SPEC.md and DECISIONS.md drafted.
- [x] Phase 2: standards v2.0.1 installed, protected paths set, four owner-only skills added.
- [x] Phase 3: repo-wide style cleanup (`06a4f06`).
- [x] Phase 4: fixed `evid-collect --seed` being discarded (`1d98db8`), added property and pinned-behavior tests (200 passing), fixed three ruff findings, gate is `make agent-check`.

## Open questions for the owner

- `.env` holds a Tavily key. It was never tracked or in history, so no rotation is needed, but `artifacts/cache/` holds fetched web content and stays local.
- Collection runs made with `evid-collect` before `1d98db8` replayed the same random stream whatever `--seed` was. Decide whether any imitation data or results that depended on them need regenerating.
- Reported results live only in `artifacts/experiments/`, which is gitignored. They cannot be reproduced from the repo until a results table is committed.
