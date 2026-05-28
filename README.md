# EvidenceRL

**EvidenceRL** is a reinforcement learning framework that trains agents to verify scientific and factual claims through iterative evidence gathering and debate-style reasoning. The agent operates in a custom Gym-style environment where each episode presents a claim, a pool of evidence, and a structured action space for building arguments — then receives shaped rewards based on the quality of its reasoning process and the accuracy of its final judgment. The core contribution is the RL training loop itself: a principled formulation of claim verification as a sequential decision-making problem, with support for multi-armed bandits, policy gradient, and PPO out of the box.

---

## Architecture

### RL Environment (`ClaimEnv`)

Each episode initializes with a claim and an evidence pool. The agent navigates a five-action space:

| Action | Description |
|---|---|
| `SELECT` | Pull a piece of evidence into the active context |
| `REMOVE` | Drop evidence from the active context |
| `SUPPORT` | Generate an argument in favor of the claim |
| `CONTRADICT` | Generate a counter-argument against the claim |
| `FINALIZE` | Commit to a final credibility judgment |

State at each step includes the claim, the full evidence pool, the currently selected evidence, and the full debate history accumulated via `SUPPORT`/`CONTRADICT` actions.

### Debate Loop

The `SUPPORT`/`CONTRADICT` cycle is the central reasoning mechanism. Rather than issuing a single judgment from a static context, the agent constructs an explicit argument trace — alternating between building the case for and against the claim — before calling `FINALIZE`. This debate history is passed to the LLM judge at evaluation time, making the agent's reasoning process legible and directly optimizable via reward shaping.

---

## Reward Design

Rewards are computed at two levels:

### Step Rewards

Issued at each action to shape the learning signal mid-episode:

- **Positive:** selecting relevant, non-redundant evidence; generating coherent supporting or contradicting arguments
- **Negative:** redundant selections, irrelevant evidence, excessive steps without progress

Argument-generating actions (`SUPPORT`, `CONTRADICT`) receive an additional LLM-shaped reward: a lightweight LLM judge scores the accumulated debate history at each step, and a weighted fraction of that score is folded into the step reward. This grounds intermediate shaping in argument quality rather than heuristics alone.

### Final Reward

Issued on `FINALIZE`, based on:

- Alignment with the ground-truth label
- Logical consistency of the debate trace
- Evidence utilization quality (coverage, relevance, non-redundancy)

The final reward is the primary training signal; step rewards are shaped to encourage efficient evidence selection and coherent argument construction on the path to it.

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

Configure via `config.py`:

```python
RL_METHOD = "ppo"  # options: "bandit", "pg", "ppo"
NUM_EPISODES = 100
```

---

## Installation

```bash
git clone https://github.com/nickmelamed/evid_rl.git
cd evid_rl

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

---

## Usage

### Run a Single Episode

```bash
python run_episode.py
```

Outputs actions taken, intermediate rewards, and the final decision.

### Train the Agent

```bash
python train_rl.py
```

Training logs to `logs/experiment_results.csv` automatically, capturing episode rewards, final scores, and action distributions.

### Visualize a Run

```bash
python visualize.py --file logs/<experiment_name>.csv
```

Renders learning curve, rolling average reward, and action distribution trends over training.

### Compare Experiments

```bash
python compare_experiments.py \
    --files logs/ppo_run.csv logs/pg_run.csv logs/bandit_run.csv
```

Overlays reward curves across runs for method comparison and hyperparameter analysis, with convergence speed and final performance breakdowns.

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

- **Learned reward models:** replace the heuristic + LLM-judge reward function with a trained reward model fine-tuned on human preference data over argument quality
- **Multi-agent debate:** pit two independent agents against each other — one tasked with support, one with contradiction — with a separate arbiter issuing the final reward signal
- **Domain expansion:** extend beyond scientific claims to regulatory filings, clinical trial reports, and policy documents, with domain-specific evidence retrievers and reward calibration
