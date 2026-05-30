"""
CLI entry point: evid-collect

Collects labeled trajectories for imitation learning / offline RL.
Three modes:
  llm_annotator   — strong LLM selects actions (AnnotatorClient)
  reward_filtered — random rollouts, keep top-k% by reward
  best_rollouts   — load existing JSONL files from logs/, filter by min-reward

Output: newline-delimited JSON, one trajectory object per line.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import random

import numpy as np

from evid_rl_env.data.dataset import load_dataset
from evid_rl_env.environment.environment import ClaimEnv
from evid_rl_env.environment.actions import ACTIONS
from evid_rl_env.agent.baseline import _build_payload, _state_summary

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

N_ACTIONS = len(ACTIONS)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _load_train_split(dataset_path: str | None) -> list:
    """Load the training split (same 80/20 seed-42 split as train.py)."""
    if dataset_path:
        with open(dataset_path) as f:
            return json.load(f)

    dataset = load_dataset()
    random.seed(42)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    split = int(0.8 * len(dataset))
    return [dataset[i] for i in indices[:split]]


# ---------------------------------------------------------------------------
# Episode collection
# ---------------------------------------------------------------------------

def _collect_episode(env: ClaimEnv, action_fn, annotator_model: str, mode: str) -> dict:
    """
    Run one episode and return a trajectory dict in the canonical JSONL format.
    action_fn(obs_str, state) -> int
    """
    state = env.reset()
    claim = state.claim
    done = False
    total_reward = 0.0
    steps = []

    while not done:
        obs = _state_summary(state)
        action_idx = int(np.clip(action_fn(obs, state), 0, N_ACTIONS - 1))
        action = ACTIONS[action_idx]
        payload = _build_payload(action, state)

        next_state, reward, done, _ = env.step(action, payload)
        total_reward += reward
        next_obs = _state_summary(next_state)

        steps.append({
            "obs": obs,
            "action": action_idx,
            "reward": float(reward),
            "next_obs": next_obs,
            "done": done,
        })
        state = next_state

    return {
        "claim": claim,
        "annotator_model": annotator_model,
        "mode": mode,
        "total_reward": float(total_reward),
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# Mode: llm_annotator
# ---------------------------------------------------------------------------

def collect_llm_annotator(env: ClaimEnv, annotator, n_episodes: int, model_name: str) -> list:
    trajectories = []
    for ep in range(n_episodes):
        def action_fn(obs, state):
            return annotator.select_action(obs, list(ACTIONS))

        traj = _collect_episode(env, action_fn, annotator_model=model_name, mode="llm_annotator")
        trajectories.append(traj)
        logger.info("llm_annotator ep %d/%d | reward=%.3f", ep + 1, n_episodes, traj["total_reward"])
    return trajectories


# ---------------------------------------------------------------------------
# Mode: reward_filtered
# ---------------------------------------------------------------------------

def collect_reward_filtered(env: ClaimEnv, n_episodes: int, top_k_percent: float) -> list:
    all_trajectories = []
    for ep in range(n_episodes):
        def action_fn(obs, state):
            return random.randint(0, N_ACTIONS - 1)

        traj = _collect_episode(env, action_fn, annotator_model="random", mode="reward_filtered")
        all_trajectories.append(traj)

    all_trajectories.sort(key=lambda t: t["total_reward"], reverse=True)
    keep = max(1, int(len(all_trajectories) * top_k_percent / 100.0))
    filtered = all_trajectories[:keep]
    logger.info(
        "reward_filtered: kept %d/%d episodes (top %.0f%%)",
        len(filtered), n_episodes, top_k_percent,
    )
    return filtered


# ---------------------------------------------------------------------------
# Mode: best_rollouts
# ---------------------------------------------------------------------------

def collect_best_rollouts(min_reward: float, logs_dir: str = "logs") -> list:
    pattern = os.path.join(logs_dir, "**", "*.jsonl")
    jsonl_files = glob.glob(pattern, recursive=True)
    if not jsonl_files:
        # also try non-recursive in case logs_dir is flat
        jsonl_files = glob.glob(os.path.join(logs_dir, "*.jsonl"))

    if not jsonl_files:
        logger.warning("best_rollouts: no .jsonl files found in '%s'", logs_dir)
        return []

    selected = []
    for path in jsonl_files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("total_reward", float("-inf")) >= min_reward:
                    # Stamp mode for provenance; preserve original annotator_model
                    rec["mode"] = "best_rollouts"
                    selected.append(rec)

    logger.info(
        "best_rollouts: %d trajectories with total_reward >= %.3f from %d files",
        len(selected), min_reward, len(jsonl_files),
    )
    return selected


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _write_jsonl(trajectories: list, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for traj in trajectories:
            f.write(json.dumps(traj) + "\n")


def _print_summary(trajectories: list, output_path: str) -> None:
    if not trajectories:
        print("Saved 0 trajectories")
        return
    rewards = [t["total_reward"] for t in trajectories]
    print(
        f"Saved {len(trajectories)} trajectories | "
        f"Mean reward: {np.mean(rewards):.3f} | "
        f"Min: {np.min(rewards):.3f} | "
        f"Max: {np.max(rewards):.3f}"
    )
    print(f"Output: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect labeled trajectories for imitation / offline RL."
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to a JSON dataset file. Defaults to the 80/20 training split.",
    )
    parser.add_argument(
        "--mode",
        choices=["llm_annotator", "reward_filtered", "best_rollouts"],
        default="llm_annotator",
    )
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument(
        "--top-k-percent",
        type=float,
        default=20.0,
        help="Percentage of episodes to keep in reward_filtered mode.",
    )
    parser.add_argument(
        "--min-reward",
        type=float,
        default=0.5,
        help="Minimum total_reward threshold for best_rollouts mode.",
    )
    parser.add_argument(
        "--annotator-model",
        default="claude-opus-4-5",
        help="Anthropic model string for llm_annotator mode.",
    )
    parser.add_argument(
        "--output",
        default="data/trajectories.jsonl",
        help="Output JSONL path.",
    )
    args = parser.parse_args()

    train_dataset = _load_train_split(args.dataset)
    logger.info("Dataset: %d samples", len(train_dataset))

    if args.mode == "llm_annotator":
        from evid_rl_env.agent.llm_client import AnnotatorClient
        annotator = AnnotatorClient(model=args.annotator_model)
        env = ClaimEnv(train_dataset)
        trajectories = collect_llm_annotator(
            env, annotator, args.n_episodes, args.annotator_model
        )

    elif args.mode == "reward_filtered":
        env = ClaimEnv(train_dataset)
        trajectories = collect_reward_filtered(env, args.n_episodes, args.top_k_percent)

    elif args.mode == "best_rollouts":
        trajectories = collect_best_rollouts(args.min_reward)

    _write_jsonl(trajectories, args.output)
    _print_summary(trajectories, args.output)


if __name__ == "__main__":
    main()
