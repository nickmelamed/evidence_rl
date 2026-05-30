# EvidenceRL

**EvidenceRL** is a reinforcement learning framework that trains agents to verify scientific and factual claims through iterative evidence gathering and debate-style reasoning. The agent operates in a custom Gym-style environment where each episode presents a claim, a pool of evidence, and a structured action space for building arguments — then receives shaped rewards based on the quality of its reasoning process and the accuracy of its final judgment. The core contribution is the RL training loop itself: a principled formulation of claim verification as a sequential decision-making problem, with support for multi-armed bandits, policy gradient, and PPO out of the box.

---

## Architecture

### RL Environment (`ClaimEnv`)

Each episode initializes with a claim and an evidence pool. The agent navigates a nine-action space:

| Action | Description |
|---|---|
| `SELECT` | Pull a piece of evidence into the active context |
| `REMOVE` | Drop evidence from the active context |
| `SUPPORT` | Generate an argument in favor of the claim |
| `CONTRADICT` | Generate a counter-argument against the claim |
| `CONCEDE` | Acknowledge a weakness in the current argument |
| `QUERY` | Issue a follow-up Tavily search to expand the evidence pool (budget: 2 per episode) |
| `RERANK` | Reorder selected evidence by relevance |
| `SUMMARIZE` | Compress selected evidence into a summary appended to the debate history |
| `FINALIZE` | Commit to a final credibility judgment |

State at each step includes the claim, the full evidence pool, the currently selected evidence, the debate history accumulated via `SUPPORT`/`CONTRADICT`/`CONCEDE`/`SUMMARIZE` actions, and a running LLM judge score. Episodes terminate on `FINALIZE` or after 10 steps.

### Debate Loop

The `SUPPORT`/`CONTRADICT` cycle is the central reasoning mechanism. Rather than issuing a single judgment from a static context, the agent constructs an explicit argument trace — alternating between building the case for and against the claim — before calling `FINALIZE`. This debate history is passed to the LLM judge at evaluation time, making the agent's reasoning process legible and directly optimizable via reward shaping.

---

## Reward Design

Rewards are computed at two levels:

### Step Rewards

Issued at each action to shape the learning signal mid-episode:

| Action | Reward |
|---|---|
| `SELECT` (new evidence) | +0.10 + 0.05 diversity bonus |
| `SELECT` (duplicate) | 0 |
| `REMOVE` | −0.05 |
| `SUPPORT` | +0.05 + 0.15 × ΔLLM |
| `CONTRADICT` | +0.05 + 0.15 × ΔLLM |
| `CONCEDE` | +0.05 + 0.10 × ΔLLM |
| `QUERY` (within budget) | +0.05 × new documents retrieved |
| `QUERY` (over budget) | −0.10 |
| `RERANK` | +0.02 |
| `SUMMARIZE` (non-empty) | +0.05 |
| Step-limit termination | −0.20 |

ΔLLM is the change in LLM judge score relative to the previous judged step. The judge is called at most once every two steps to limit inference cost; between calls the previous score is reused. A potential-based shaping term — `0.1 × (0.99 × Φ(s') − Φ(s))`, where Φ is the current judge score — is added to every step reward, grounding dense shaping in actual argument quality.

### Final Reward

Issued on `FINALIZE` as a weighted blend of a heuristic base reward and the LLM judge reward:

```
reward = 0.70 × base_reward + 0.30 × llm_reward
```

Guards applied before the blend:

- **−1.0** if no evidence has been selected
- **−0.5** if fewer than 3 steps were taken (premature finalization)
- **−0.3** if the debate history is empty (no arguments generated)
- **+0.2 × |selected_evidence|** evidence utilization bonus

**`base_reward`** is a weighted sum of evidence quality metrics (clipped to [0, 1]):

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

**`llm_reward`** is produced by the LLM judge (see LLM Judge Metrics below).

---

## Evidence Pipeline

Each episode's evidence pool is grounded in real retrieved documents via **Tavily** search — no vector database, no pre-indexed corpus. At episode initialization, EvidenceRL issues a live Tavily query keyed to the claim, retrieves a set of documents, and constructs the evidence pool from those results. This means every episode reflects the current state of the web: the agent never reasons over stale embeddings or cached corpora.

The pipeline is intentionally lightweight:

1. Claim arrives at episode reset
2. Tavily query fires; top-k results are fetched and structured as `Evidence` objects
3. Evidence pool is passed to `ClaimEnv` — no further preprocessing
4. Agent interacts with live-retrieved evidence for the full episode

This design keeps the evidence pipeline stateless and eliminates retrieval infrastructure entirely. The tradeoff is nondeterminism across runs (web content changes), which is acceptable in a training setting where diversity of evidence is a feature, not a bug.

---

## RL Strategies

EvidenceRL supports three training strategies, switchable via a single config flag:

### Multi-Armed Bandit

Action selection modeled as a bandit problem. No temporal credit assignment — useful as a baseline to verify that the reward signal is learnable at all.

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

This installs the `evid_rl_env` package and registers the CLI entry points: `evid-train`, `evid-eval`, `evid-collect`, `evid-migrate`, `plot-exp`, `compare-exp`, and `run-episode`.

---

## Configuration

All training hyperparameters live in `configs/`. The base defaults are in `configs/base.yaml`; per-algorithm overrides are in `configs/ppo_baseline.yaml`, `configs/pg_baseline.yaml`, and `configs/bandit_baseline.yaml`.

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
  entropy_coef: 0.01
  value_coef: 0.05
  gamma: 0.99
  ppo_epochs: 4
  gae_lambda: 0.95
  actor_model: "google/gemma-2-2b-it"
  judge_model: "Qwen/Qwen2.5-1.5B-Instruct"
```

Alternatively, override individual settings via the `BaseConfig` / `PPOConfig` / `PGConfig` / `BanditConfig` classes in `src/evid_rl_env/agent/config.py`.

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

**Available baselines:**

| Baseline | Description |
|---|---|
| `random` | Uniformly random action selection |
| `majority` | Always predicts the majority-class label |
| `greedy_llm` | Single-shot LLM judgment with no debate |
| `fewshot_k3` | Few-shot LLM with 3 training examples |
| `fewshot_k5` | Few-shot LLM with 5 training examples |
| `best_of_5` | Best of 5 independent LLM samples |
| `imitation` | Policy cloned from collected expert trajectories (requires `--trajectories`) |

The table printed to stdout lists each baseline's mean ± std reward relative to `greedy_llm`, followed by the RL policy result:

```
[Eval — 50 episodes | checkpoint: artifacts/experiments/ppo_run/policy.npz]
  random          0.21 ± 0.08  Δ -0.31
  greedy_llm      0.52 ± 0.11  (baseline)
  fewshot_k3      0.58 ± 0.09  Δ +0.06
  best_of_5       0.61 ± 0.10  Δ +0.09
  RL (greedy)     0.67 ± 0.08  Δ +0.15  ← target

PASS: RL (0.670) > greedy_llm (0.520)
```

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

**Live monitoring:** enable the "Live Monitoring" toggle in the sidebar to auto-refresh at a configurable interval (1–10 seconds) while training runs.

**LLM judge metrics tracked per step:**

| Metric | Direction | Description |
|---|---|---|
| `LCS` | ↑ higher is better | Logical consistency — argument is internally coherent |
| `ESS` | ↑ higher is better | Evidence support — reasoning is grounded in selected evidence |
| `GRS` | ↓ lower is better | Grounding risk — claims introduced not present in evidence |
| `COMP` | ↑ higher is better | Completeness — all key aspects of the claim are addressed |
| `BIAS` | ↓ lower is better | Selective citation bias — only supporting evidence cited, contradictions ignored |
| `confidence` | — | Judge's confidence in the above scores |

Score-to-reward conversion: `0.30 × LCS + 0.25 × ESS + 0.20 × COMP − 0.25 × GRS − 0.15 × BIAS`. When judge confidence is below 0.4, the reward is blended 50/50 toward the neutral value of 0.5. Scores are cached by content hash to avoid redundant inference across episodes.

---

## Example Episode

1. Environment initializes with claim: *"Statins reduce cardiovascular mortality in high-risk patients"*
2. Tavily retrieves live evidence documents; pool is constructed
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

- **Learned reward models:** replace the heuristic base reward with a trained reward model fine-tuned on human preference data over argument quality, making the reward signal less sensitive to brittle heuristics like adversarial contamination labels
- **Embedding-based reranking:** `RERANK` currently orders selected evidence by text length as a proxy for relevance; replace with embedding similarity to the claim once the state encoder is wired into the action handler
- **Re-annotation pipeline:** `evid-eval --annotator-model` is wired for post-hoc re-annotation of imitation trajectories but not yet connected to a re-scoring workflow; completing this would enable iterative dataset improvement without full recollection
- **Multi-agent debate:** pit two independent agents against each other — one constrained to support, one to contradiction — with a separate arbiter issuing the final reward signal; this separates role from policy and eliminates the need for a single agent to self-regulate debate balance
- **Domain expansion:** extend beyond scientific claims to regulatory filings, clinical trial reports, and policy documents, with domain-specific evidence retrievers and reward calibration for each domain's ground-truth structure
