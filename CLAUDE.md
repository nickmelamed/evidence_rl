# EvidenceRL

EvidenceRL trains agents to verify scientific claims. The agent works in a Gym-style environment (`ClaimEnv`), gathers and argues over evidence, and is rewarded by heuristic evidence metrics plus an LLM judge. The repo has two parts: the RL framework (bandit, REINFORCE, PPO) and a study of whether judge architectures (single, debate, ensemble, escalating) track an independent gold judge. Results must come from a real run and be reproducible.

The full design lives in docs/SPEC.md. Read the relevant section before changing anything it covers. Current status and next steps are in PROGRESS.md. Design decisions and their reasons are in docs/DECISIONS.md. The README is older than the code, so trust the code and SPEC where they differ.

## Non-negotiable rules

1. Eval claims never leak into training, the curriculum, few-shot examples, or imitation data. The split is the seed-42 80/20 shuffle of `seed_claims.json`. Do not write a new copy of it.
2. Gold judges (`GoldEvaluator`) never appear in a training reward or training config.
3. Claim and evidence labels are used only for reward and evaluation, never in observations, prompts, or the state encoder.
4. Every reported number comes from a real run in `artifacts/experiments/` or a committed results table. Never type a result by hand.
5. Randomness is seeded and the same seed gives the same result. Restore global RNG state after temporary seeding. Compare methods on identical episodes.
6. Never commit `.env`, API keys, or cache files. `.env.example` holds names only.
7. Never weaken a test, lint rule, or check to make it pass.
8. Ask before changing anything in `.claude/protected-paths`, the evaluation design, reward weights, dependencies, CI, or hooks.
9. If you find a real bug while doing something else, stop and tell me. Do not fix it in passing.

## Commands

```bash
pip install -e .[dev]   # install, inside the ev_rl venv (source ev_rl/bin/activate)
make test               # pytest, about 20s, no network or models
make lint               # ruff check .
make agent-check        # lint, fast tests, style check
make ci                 # what CI runs
```

Training and evaluation need models and a Tavily key and are slow. Do not run `evid-train`, `evid-eval`, or `evid-gold-eval` without asking.

## Where things live

- `src/evid_rl_env/`: `environment/` (ClaimEnv), `judge/` (judges, reward, metrics), `agent/` (policies, trainers, baselines, evaluators), `cli/`, `data/`, `utils/`.
- `configs/`: YAML hyperparameters, the single source of truth for tuning.
- `artifacts/`: gitignored runs and sqlite caches. `dashboard/app.py`: Streamlit view of runs.
- `tests/`: synthetic fixtures with mocked judges. `docs/`: SPEC and DECISIONS.
- `.claude/rules/`: path-scoped rules that load when you touch those files.

## How to work here

- Plan before multi-file changes. Write the plan down and wait for approval when the task spans several modules or touches the spec.
- A task is done when the Stop hook's checks pass and you have shown the output. Show commands and results, not claims.
- Commit in small atomic Conventional Commits (`type(scope): subject`). Code and its tests go in the same commit.
- You may branch and commit locally. Ask before pushing, opening or merging pull requests, tagging, or changing dependencies.
- If you repeat a multi-step procedure, or I give you the same instructions twice, propose a skill in `.claude/skills/` and wait for approval.
