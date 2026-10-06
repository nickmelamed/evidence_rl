# EvidenceRL

**EvidenceRL** is a reinforcement learning framework that trains agents to verify scientific and factual claims through iterative evidence gathering and debate-style reasoning. The agent operates in a custom Gym-style environment where each episode presents a claim, a pool of evidence, and a structured action space for building arguments, then receives shaped rewards based on the quality of its reasoning process and the accuracy of its final judgment. The core contribution is the RL training loop itself: a principled formulation of claim verification as a sequential decision-making problem, with support for multi-armed bandits, policy gradient, and PPO out of the box.

---

## Architecture

### RL Environment (`ClaimEnv`)

Each episode samples a claim and an evidence pool. The agent chooses from thirteen actions (`environment/actions.py`):

| Action | Description |
|---|---|
| `SELECT` | Pull a piece of evidence into the active context |
| `REMOVE` | Drop evidence from the active context |
| `SUPPORT` | Generate an argument in favor of the claim |
| `CONTRADICT` | Generate a counter-argument against the claim |
| `CONCEDE` | Acknowledge a weakness in the current argument |
| `HEDGE` | Add a qualified statement to the debate history |
| `CHALLENGE_EVIDENCE` | Argue that a specific evidence item is unreliable |
| `QUERY` | Issue a follow-up Tavily search to expand the evidence pool |
| `REQUEST_CLARIFICATION` | Like `QUERY`, and also records the question in the debate history |
| `RERANK` | Reorder selected evidence by similarity to the claim |
| `SUMMARIZE` | Compress selected evidence into a summary appended to the debate history |
| `ASSIGN_CONFIDENCE` | Set the confidence used when the episode is scored |
| `FINALIZE` | Commit to a final judgment |

`QUERY` and `REQUEST_CLARIFICATION` share a budget of two per episode. State includes the claim, the evidence pool, the selected evidence, the debate history and the last judge score. Episodes end on `FINALIZE` (with at least one selected item) or after 10 steps.

### Debate Loop

The `SUPPORT`/`CONTRADICT` cycle is the central reasoning mechanism. Rather than issuing a single judgment from a static context, the agent constructs an explicit argument trace, alternating between building the case for and against the claim, before calling `FINALIZE`. This debate history is passed to the LLM judge at evaluation time, making the agent's reasoning process legible and directly optimizable via reward shaping.

---

## Reward Design

Rewards are computed at two levels:

### Step Rewards

Issued at each action to shape the learning signal mid-episode:

| Action | Reward |
|---|---|
| `SELECT` (new evidence) | +0.15 support, +0.10 contradict, 0 neutral, −0.15 adversarial, plus a 0.05 bonus for evidence not selected before | <!-- numbers: ok -->
| `SELECT` (duplicate) | −0.1 | <!-- numbers: ok -->
| `REMOVE` | −0.05 | <!-- numbers: ok -->
| `SUPPORT`, `CONTRADICT`, `CHALLENGE_EVIDENCE` | +0.05 + 0.15 × ΔLLM | <!-- numbers: ok -->
| `CONCEDE`, `HEDGE` | +0.05 + 0.10 × ΔLLM | <!-- numbers: ok -->
| `CHALLENGE_EVIDENCE` extra | +0.10 for an adversarial target, −0.05 for a supporting one | <!-- numbers: ok -->
| `QUERY`, `REQUEST_CLARIFICATION` (within budget) | min(0.15, 0.05 × useful new documents) | <!-- numbers: ok -->
| `QUERY`, `REQUEST_CLARIFICATION` (over budget) | −0.10 | <!-- numbers: ok -->
| `RERANK` | +0.02 | <!-- numbers: ok -->
| `SUMMARIZE` (non-empty) | +0.05 | <!-- numbers: ok -->
| `ASSIGN_CONFIDENCE` | +0.05 | <!-- numbers: ok -->
| Step-limit termination | −0.20 | <!-- numbers: ok -->

ΔLLM is the change in LLM judge score relative to the previous judged step. The judge is called at most once every two steps to limit inference cost, and between calls the previous score is reused. A potential-based shaping term, `0.1 × (0.99 × Φ(s') − Φ(s))`, is added to every step reward. Φ is the current judge score and is 0 at terminal states.

### Final Reward

Issued on `FINALIZE`:

- **−1.0** and the episode ends if no evidence has been selected.
- **−0.5** and the episode continues if fewer than 3 steps were taken.
- Otherwise `reward = base_reward − 0.3 (if the debate history is empty) + min(0.3, 0.1 × useful selected evidence)`.

`base_reward` is a weighted sum of evidence quality metrics, clipped to [−1, 1], and is 0 when the reasoning is empty:

```
base_reward = 0.40 × F1
            + 0.20 × CA
            − 0.15 × AC
            − 0.15 × uncertainty_penalty
            + 0.10 × llm_reward
            − overselection_penalty
```

| Metric | Description |
|---|---|
| F1 | Harmonic mean of precision and recall over supporting evidence |
| CA (Contradiction Acknowledgment) | Fraction of contradicting evidence in the pool that was selected |
| AC (Adversarial Contamination) | Fraction of selected evidence labeled adversarial |
| Uncertainty penalty | `|confidence − true_score|` |
| Overselection penalty | `0.04 × max(0, |selected| − 5)` |

Confidence is the agent's `ASSIGN_CONFIDENCE` value, or `min(1, selected / 3)` if it never assigned one. `true_score` is the claim's label.

**`llm_reward`** is produced by the LLM judge (see LLM Judge Metrics below).

---

## Evidence Pipeline

Claims in `seed_claims.json` carry gold SciFact evidence with human labels, and the environment uses it directly. For claims without it, the evidence pool is built from live **Tavily** search results, with no vector database and no pre-indexed corpus.

For a live-search claim the pipeline is:

1. The claim arrives at episode reset.
2. A Tavily query fires and the top-k results are structured as `Evidence` objects.
3. Each document is labeled by `judge.evidence_labeler.EvidenceLabeler`, a cached LLM call that reuses the judge model. It records the document's stance (`support`, `contradict` or `neutral`) and whether it looks adversarial. These labels feed the F1, CA and AC terms of `base_reward`.
4. The pool is passed to `ClaimEnv` and the agent works with it for the episode.

Live search keeps retrieval infrastructure minimal. The tradeoff is that a fresh checkout with a cold cache sees different evidence than a previous run (web content changes). The fetch cache (`artifacts/cache/fetch_cache.sqlite3`) makes results stable on one machine once warmed. They are not portable across machines. For results you need to reproduce exactly (e.g. in a report), use `evid-snapshot` to export a portable JSON snapshot, and pass `--evidence-snapshot` to `evid-train`/`evid-eval` to load it. The snapshot bypasses both the live Tavily call and the local sqlite cache for any claim it covers.

---

## RL Strategies

EvidenceRL supports three training strategies, switchable via a single config flag:

### Multi-Armed Bandit

Action selection modeled as a bandit problem. No temporal credit assignment. It is useful as a baseline to verify that the reward signal is learnable at all.

### Policy Gradient (REINFORCE)

Full episode rollouts with Monte Carlo return estimates. Learns a policy over the action space directly from cumulative episodic reward.

### Proximal Policy Optimization (PPO)

Clipped surrogate objective with value function baseline. Stable under the high-variance reward signal that comes from LLM-in-the-loop shaping. Recommended for serious training runs.

Configure via `configs/`:

```bash
evid-train --method ppo --episodes 100   # options: bandit, pg, ppo
```

Or override hyperparameters directly in `configs/ppo_baseline.yaml` (or `pg_baseline.yaml` / `bandit_baseline.yaml`).

---

## Installation

```bash
git clone https://github.com/nickmelamed/evid_rl.git
cd evid_rl

python -m venv ev_rl
source ev_rl/bin/activate

pip install -e .
```

This installs the `evid_rl_env` package and registers the CLI entry points: `evid-train`, `evid-eval`, `evid-collect`, `evid-migrate`, `evid-snapshot`, `plot-exp`, `compare-exp`, and `run-episode`.

For running the test suite and linter, install the `dev` extra instead: `pip install -e .[dev]`.

---

## Testing

```bash
make test   # or: pytest
make lint   # or: ruff check .
```

The test suite mocks every LLM/judge boundary (`ClaimEnv` accepts an injected
`llm_judge=`, and the baseline classes take a duck-typed `llm_client`), so it
never downloads or runs the actual gemma/qwen generation models and needs no
API keys or network access. A GitHub Actions workflow (`.github/workflows/ci.yml`)
runs both on every push and pull request.

---

## Configuration

All training hyperparameters live in `configs/`. The base defaults are in `configs/base.yaml`. Per-algorithm overrides are in `configs/ppo_baseline.yaml`, `configs/pg_baseline.yaml`, and `configs/bandit_baseline.yaml`.

```yaml
# configs/base.yaml
seed: 42
default_annotator_model: claude-opus-4-5
eval_every: 10          # run held-out eval every N episodes
```

```yaml
# configs/ppo_baseline.yaml
algo: ppo
rl:
  lr: 0.001
  clip: 0.2
  entropy_coef: 0.05  # raised from 0.01 — too weak to prevent collapse on short runs
  value_coef: 0.05
  gamma: 0.99
  ppo_epochs: 2        # lowered from 4 — K=4 over-commits weights after only a few episodes
  gae_lambda: 0.95
  actor_model: "google/gemma-2-2b-it"
  judge_model: "Qwen/Qwen2.5-1.5B-Instruct"
```

`src/evid_rl_env/agent/config.py`'s `BaseConfig`/`PPOConfig`/`PGConfig`/`BanditConfig` classes define the config *schema* and hold real defaults only for cross-cutting fields that aren't per-run tuning knobs (model choices, seed). Algorithm-specific RL hyperparameters (`lr`, `clip`, `entropy_coef`, `gamma`, `alpha`, ...) are left unset there. Set them in `configs/*_baseline.yaml` (or point `--config` at a new file) rather than in the Python class.

---

## Usage

### Run a Single Episode

```bash
run-episode
# or: make episode
```

Prints actions taken, intermediate rewards, and the final decision for one episode.

### Train the Agent

```bash
evid-train --method ppo --episodes 100
# or via make:
make train-ppo
make train-pg
make train-bandit
```

Training artifacts are written to `artifacts/experiments/<run_name>/`:

| File | Contents |
|---|---|
| `metrics.csv` | Per-episode reward, entropy, token usage, eval snapshots |
| `config.json` | Full config used for the run |
| `policy.npz` | Saved policy checkpoint |
| `trajectories/episode_<N>.json` | Step-by-step trajectory for each episode |

---

## Evaluation

Evaluation runs the trained RL policy against a suite of baselines on the held-out 20% split (deterministic seed-42 80/20 split of `seed_claims.json`).

### Quick eval

```bash
make eval checkpoint=artifacts/experiments/<run_name>/policy.npz
# runs: random, greedy_llm, fewshot_k3, best_of_5 — 50 episodes
```

### Full eval

```bash
make eval-full checkpoint=artifacts/experiments/<run_name>/policy.npz
# runs all baselines over 100 episodes
```

### CI gate

```bash
make eval-ci checkpoint=artifacts/experiments/<run_name>/policy.npz
# exits 0 if RL beats greedy_llm, exits 1 otherwise
```

### Eval with imitation baseline

```bash
make eval-imitation checkpoint=<path> trajectories=<path>
```

### Direct CLI

```bash
evid-eval \
  --checkpoint artifacts/experiments/<run_name>/policy.npz \
  --baselines random,greedy_llm,fewshot_k3,best_of_5 \
  --n-episodes 50 \
  --seed 0
```

The available baselines are:

| Baseline | Description |
|---|---|
| `random` | Uniformly random action selection |
| `majority` | Always predicts the majority-class label |
| `greedy_llm` | Single-shot LLM judgment with no debate |
| `fewshot_k3` | Few-shot LLM with 3 training examples |
| `fewshot_k5` | Few-shot LLM with 5 training examples |
| `best_of_5` | Best of 5 independent LLM samples |
| `imitation` | Policy cloned from collected expert trajectories (requires `--trajectories`) |

`evid-eval` prints a table of mean ± std reward for each baseline and the RL policy, and writes `eval_results.json` to the run folder. The exit code is 0 if the RL policy beats `greedy_llm` and 1 if it does not.

---

## Results

The tables in `results/tables/` are generated from the run folders in `artifacts/experiments/` by `make results`. Every row names its source run, and `make numbers` checks that the numbers quoted in this README match them. Nothing here is typed in by hand.

### RL policy against baselines (`eval_baselines.csv`)

Three runs from 2026-05-31 were evaluated on the held-out split for 50 episodes each. Rewards are mean ± std per episode.

| Run | RL policy | `random` | `greedy_llm` |
|---|---|---|---|
| `bandit_run_20260531_171748` | 0.781 ± 0.4906 | −0.1769 ± 0.7856 | 0.6284 ± 0.5036 |
| `pg_run_20260531_164320` | 0.072 ± 0.1808 | −0.1769 ± 0.7856 | 0.69 ± 0.5684 |
| `ppo_run_20260531_155047` | 0.16 ± 0.3303 | −0.1769 ± 0.7856 | 0.6203 ± 0.5161 |

Only the bandit run beat `greedy_llm`. These runs predate several reward and judge changes, so they should be rerun before drawing conclusions.

### Gold judge agreement (`gold_eval_last_round.csv`)

`evid-gold-eval` re-scores each run's final reasoning with a held-out judge. The table lists the last logged round for each run. `n_scored` is the number of episodes that reached `FINALIZE` and could be scored, out of `n_episodes`. Correlations from one to five scored episodes say very little, and an empty cell means there were too few scored episodes to compute one.

The one run with a usable sample is `comparison_escalation_20260712_001008`, with 5 scored episodes out of 25, a proxy-gold correlation of 0.7298, and an escalation rate of 0.8.

Training reward per run is in `training_summary.csv`.

---

## Trajectory Collection (Imitation Learning)

To bootstrap the imitation baseline or warm-start training with expert demonstrations, use `evid-collect`:

```bash
# Collect with a strong LLM annotator (default: claude-opus-4-5)
evid-collect --mode llm_annotator --n-episodes 50 --output data/trajectories.jsonl

# Collect random rollouts and keep top 20% by reward
evid-collect --mode reward_filtered --n-episodes 200 --top-k-percent 20

# Filter existing JSONL logs by minimum reward
evid-collect --mode best_rollouts --min-reward 0.5

# or via make:
make collect
make collect-n n=100
```

Output is a JSONL file (one trajectory per line) compatible with the `imitation` baseline.

To migrate trajectory files after format changes:

```bash
make migrate
```

---

## Analysis

### Plot a single experiment

```bash
plot-exp --path artifacts/experiments/<run_name>
# saves reward_curve.png to the experiment directory
# or: make plot path=artifacts/experiments/<run_name>
```

### Compare experiments

```bash
compare-exp --paths artifacts/experiments/ppo_run artifacts/experiments/pg_run
# or: make compare paths="artifacts/experiments/ppo_run artifacts/experiments/pg_run"
```

Overlays smoothed reward curves across runs for method comparison.

---

## Dashboard

The Streamlit dashboard provides live monitoring during training and post-hoc experiment analysis.

```bash
make dashboard
# or: streamlit run dashboard/app.py
```

The dashboard reads from `artifacts/experiments/` and offers four views:

| Tab | Contents |
|---|---|
| **Single Run** | Reward curve (raw + smoothed), LLM judge scores (LCS / ESS / HRS), token usage, train vs eval gap, policy entropy, action distribution over time |
| **Compare** | Overlaid smoothed reward curves across selected experiments, per-experiment LLM judge score trends |
| **Episode Drilldown** | Step-by-step replay for any episode: claim, evidence pool, generated arguments, action probabilities, value estimates, advantage signal, LLM judge scores per step |
| **Config** | Full hyperparameter table for each selected experiment |

To monitor live, enable the "Live Monitoring" toggle in the sidebar to auto-refresh at a configurable interval (1–10 seconds) while training runs.

The judge tracks these metrics at every step:

| Metric | Direction | Description |
|---|---|---|
| `LCS` | ↑ higher is better | Logical consistency: argument is internally coherent |
| `ESS` | ↑ higher is better | Evidence support: reasoning is grounded in selected evidence |
| `GRS` | ↓ lower is better | Grounding risk: claims introduced not present in evidence |
| `COMP` | ↑ higher is better | Completeness: all key aspects of the claim are addressed |
| `BIAS` | ↓ lower is better | Selective citation bias: only supporting evidence cited, contradictions ignored |
| `confidence` | n/a | Judge's confidence in the above scores |

Score-to-reward conversion: `0.30 × LCS + 0.25 × ESS + 0.20 × COMP − 0.25 × GRS − 0.15 × BIAS`. When judge confidence is below 0.4, the reward is blended 50/50 toward the neutral value of 0.5. Scores are cached by content hash to avoid redundant inference across episodes.

---

## Example Episode

1. Environment initializes with claim: *"Statins reduce cardiovascular mortality in high-risk patients"*
2. Tavily retrieves live evidence documents and the pool is constructed
3. Agent iterates:
   - `SELECT` → pulls two high-relevance documents
   - `SUPPORT` → generates argument citing trial data
   - `CONTRADICT` → generates counter citing confounding study
   - `SELECT` → adds a third document addressing the confounder
   - `SUPPORT` → strengthens argument with updated evidence
   - `FINALIZE` → commits judgment
4. Reward model evaluates evidence coverage, debate coherence, and alignment with ground truth

---

## Future Work

- Learned reward models: replace the heuristic base reward with a trained reward model fine-tuned on human preference data over argument quality, making the reward signal less dependent on the `EvidenceLabeler`'s own LLM-based stance/reliability labels, which are still a heuristic proxy and not ground truth
- Re-annotation pipeline: `evid-collect --annotator-model` is wired for labeling trajectories with a strong LLM but not yet connected to a re-scoring workflow for *existing* imitation trajectories. Completing this would enable iterative dataset improvement without full recollection
- Multi-agent debate: pit two independent agents against each other, one constrained to support, one to contradiction, with a separate arbiter issuing the final reward signal. This separates role from policy and eliminates the need for a single agent to self-regulate debate balance
- Domain expansion: extend beyond scientific claims to regulatory filings, clinical trial reports, and policy documents, with domain-specific evidence retrievers and reward calibration for each domain's ground-truth structure
