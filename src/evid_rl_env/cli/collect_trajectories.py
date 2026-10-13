"""
CLI entry point: evid-collect

Collects labeled trajectories for imitation learning / offline RL.
Four modes:
  llm_annotator: strong LLM selects actions (AnnotatorClient)
  reward_filtered: random rollouts, keep top-k% by reward
  best_rollouts: load existing JSONL files from logs/, filter by min-reward
  rescore: replay the actions of an existing JSONL file under the current
    environment and reward, optionally adding a strong LLM's action per step

Output: newline-delimited JSON, one trajectory object per line.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import random
from datetime import datetime, timezone

import numpy as np

from evid_rl_env.agent.baseline import _build_payload, _state_summary
from evid_rl_env.agent.config_loader import load_base_config
from evid_rl_env.data.dataset import load_dataset, split_dataset
from evid_rl_env.environment.actions import ACTIONS
from evid_rl_env.environment.environment import ClaimEnv

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

N_ACTIONS = len(ACTIONS)

# define the set of modes that originate from training-only collection;
# best_rollouts uses this to filter out any stray non-training records
_TRAIN_COLLECTION_MODES = frozenset(
    {"llm_annotator", "reward_filtered", "best_rollouts", "rescored"}
)


# dataset helpers 

def _load_train_split(dataset_path: str | None) -> list:
    """Load the training split (same 80/20 seed-42 split as train.py)."""
    if dataset_path:
        with open(dataset_path) as f:
            return json.load(f)

    dataset = load_dataset()
    return split_dataset(dataset)[0]


# episode collection 

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
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "total_reward": float(total_reward),
        "steps": steps,
    }


# LLM annotation 

def collect_llm_annotator(env: ClaimEnv, annotator, n_episodes: int, model_name: str) -> list:
    trajectories = []
    for ep in range(n_episodes):
        def action_fn(obs, state):
            return annotator.select_action(obs, list(ACTIONS))

        traj = _collect_episode(env, action_fn, annotator_model=model_name, mode="llm_annotator")
        trajectories.append(traj)
        logger.info("llm_annotator ep %d/%d | reward=%.3f", ep + 1, n_episodes, traj["total_reward"])
    return trajectories


# Reward filtered mode 

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


# Best rollouts mode 

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
    skipped_mode = 0
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
                # only include records whose mode is a recognized training
                # collection mode; records without a mode field or with an unknown mode
                # are skipped with a warning to prevent eval-episode data leaking into
                # the imitation learning dataset
                traj_mode = rec.get("mode")
                if traj_mode not in _TRAIN_COLLECTION_MODES:
                    logger.warning(
                        "best_rollouts: skipping record with mode=%r (not a recognized "
                        "training-collection mode) in file %s",
                        traj_mode, path,
                    )
                    skipped_mode += 1
                    continue
                if rec.get("total_reward", float("-inf")) >= min_reward:
                    # Stamp mode for provenance; preserve original annotator_model
                    rec["mode"] = "best_rollouts"
                    selected.append(rec)

    if skipped_mode:
        logger.warning(
            "best_rollouts: skipped %d records with unrecognized modes across %d files",
            skipped_mode, len(jsonl_files),
        )

    logger.info(
        "best_rollouts: %d trajectories with total_reward >= %.3f from %d files",
        len(selected), min_reward, len(jsonl_files),
    )
    return selected


# Rescore mode 

def _replay_trajectory(env: ClaimEnv, rec: dict, seed_key: str, annotator=None,
                       annotator_model: str | None = None) -> dict:
    """Replay the recorded action indices of ``rec`` in ``env`` and score them again.

    Trajectories store the action index but not the payload, so the payloads are
    rebuilt. The global RNG is seeded from ``seed_key`` for the replay and
    restored afterwards, which makes the rebuilt payloads repeatable. The new
    reward is therefore the reward of a seeded replay, not of the original episode.
    """
    saved_state = random.getstate()
    random.seed(seed_key)
    try:
        state = env.reset()
        steps = []
        total_reward = 0.0
        done = False
        recorded = rec.get("steps", [])
        for step in recorded:
            if done:
                break
            obs = _state_summary(state)
            action_idx = int(np.clip(int(step["action"]), 0, N_ACTIONS - 1))
            action = ACTIONS[action_idx]
            next_state, reward, done, _ = env.step(action, _build_payload(action, state))
            total_reward += reward
            new_step = {
                "obs": obs,
                "action": action_idx,
                "reward": float(reward),
                "next_obs": _state_summary(next_state),
                "done": done,
                "source_reward": step.get("reward"),
            }
            if annotator is not None:
                new_step["annotator_action"] = int(annotator.select_action(obs, list(ACTIONS)))
            steps.append(new_step)
            state = next_state
    finally:
        random.setstate(saved_state)

    out = {
        "claim": rec["claim"],
        "annotator_model": rec.get("annotator_model"),
        "mode": "rescored",
        "source_mode": rec.get("mode"),
        "collected_at": rec.get("collected_at"),
        "rescored_at": datetime.now(timezone.utc).isoformat(),
        "source_total_reward": rec.get("total_reward"),
        "total_reward": float(total_reward),
        "steps_dropped": len(recorded) - len(steps),
        "steps": steps,
    }
    if annotator is not None:
        out["relabel_model"] = annotator_model
    return out


def rescore_trajectories(records: list, train_dataset: list, env_factory, seed: int = 0,
                         annotator=None, annotator_model: str | None = None) -> list:
    """Replay existing trajectories under the current environment and reward.

    Only records for claims in ``train_dataset`` with a training collection mode
    are kept. Anything else is skipped, so eval claims cannot enter the imitation
    data through an old file. ``env_factory`` gets a one-claim dataset and returns
    an environment.
    """
    by_claim = {s["claim"]: s for s in train_dataset}
    rescored = []
    skipped_split = skipped_mode = 0
    for i, rec in enumerate(records):
        claim = rec.get("claim")
        if claim not in by_claim:
            skipped_split += 1
            continue
        if rec.get("mode") not in _TRAIN_COLLECTION_MODES:
            skipped_mode += 1
            continue
        env = env_factory([by_claim[claim]])
        rescored.append(
            _replay_trajectory(env, rec, f"{seed}:{i}:{claim}", annotator, annotator_model)
        )
        logger.info(
            "rescore %d/%d | reward %s -> %.3f",
            i + 1, len(records), rec.get("total_reward"), rescored[-1]["total_reward"],
        )
    if skipped_split or skipped_mode:
        logger.warning(
            "rescore: skipped %d records outside the training split and %d with an "
            "unrecognized mode", skipped_split, skipped_mode,
        )
    return rescored


def _read_jsonl(path: str) -> list:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


# Output 

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


# CLI 

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
        choices=["llm_annotator", "reward_filtered", "best_rollouts", "rescore"],
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
        default=load_base_config().get("default_annotator_model", "claude-opus-4-5"),
        help="Anthropic model string for llm_annotator mode.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Trajectory JSONL to replay in rescore mode.",
    )
    parser.add_argument(
        "--relabel",
        action="store_true",
        help="In rescore mode, also ask --annotator-model for an action at each step.",
    )
    parser.add_argument(
        "--output",
        default="data/trajectories.jsonl",
        help="Output JSONL path.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for episode collection (default: 0). "
             "Does not affect the train/eval split (always seed 42).",
    )
    args = parser.parse_args()

    if args.mode == "rescore":
        if not args.input:
            parser.error("--input is required in rescore mode")
        if os.path.abspath(args.input) == os.path.abspath(args.output):
            parser.error("--output must differ from --input, rescoring never overwrites")

    random.seed(args.seed)
    np.random.seed(args.seed)

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

    elif args.mode == "rescore":
        annotator = None
        if args.relabel:
            from evid_rl_env.agent.llm_client import AnnotatorClient
            annotator = AnnotatorClient(model=args.annotator_model)
        trajectories = rescore_trajectories(
            _read_jsonl(args.input), train_dataset, lambda ds: ClaimEnv(ds),
            seed=args.seed, annotator=annotator, annotator_model=args.annotator_model,
        )

    _write_jsonl(trajectories, args.output)
    _print_summary(trajectories, args.output)


if __name__ == "__main__":
    main()
