"""
Baseline evaluators for benchmarking the RL agent.

All classes inherit from BaseEvaluator and use the shared run_episode()
helper so episode logic is never duplicated across implementations.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import random
import re
import warnings
from abc import ABC, abstractmethod

import numpy as np

from evid_rl_env.environment.environment import ClaimEnv
from evid_rl_env.environment.actions import ACTIONS, Actions

logger = logging.getLogger(__name__)

N_ACTIONS = len(ACTIONS)
ACTION_DESCRIPTIONS = "\n".join(f"{i}: {a}" for i, a in enumerate(ACTIONS))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_payload(action: str, state) -> object:
    """Minimal valid payload for each action type."""
    if action == Actions.SELECT:
        return random.choice(state.evidence_pool).id if state.evidence_pool else 0
    if action == Actions.REMOVE:
        return random.choice(state.selected_evidence).id if state.selected_evidence else 0
    if action in (Actions.SUPPORT, Actions.CONTRADICT):
        return {"argument": "argument", "evidence_ids": [], "tokens": 1}
    if action == Actions.QUERY:
        return state.claim
    if action == Actions.SUMMARIZE:
        return {"summary": "summary", "tokens": 1}
    if action == Actions.CONCEDE:
        return {"argument": "concession", "evidence_ids": [], "tokens": 1}
    return None  # RERANK, FINALIZE


def _state_summary(state) -> str:
    """Compact text description of state for LLM prompts."""
    return (
        f"Claim: {state.claim} | "
        f"Pool: {len(state.evidence_pool)} docs | "
        f"Selected: {len(state.selected_evidence)} | "
        f"Step: {state.steps_taken}/{state.max_steps} | "
        f"Debate turns: {len(state.debate_history)}"
    )


def run_episode(env: ClaimEnv, action_fn) -> tuple:
    """
    Run one full episode.

    action_fn(state) -> int   (action index)

    Returns:
        trajectory   — list of per-step dicts
        total_reward — scalar float
        llm_scores   — {LCS, ESS, GRS, COMP} lists for metric aggregation
    """
    state = env.reset()
    done = False
    total_reward = 0.0
    trajectory = []
    llm_scores = {"LCS": [], "ESS": [], "GRS": [], "COMP": []}

    while not done:
        action_idx = int(np.clip(action_fn(state), 0, N_ACTIONS - 1))
        action = ACTIONS[action_idx]
        payload = _build_payload(action, state)

        next_state, reward, done, info = env.step(action, payload)
        total_reward += reward

        step_scores = info.get("llm_scores", {})
        for key in llm_scores:
            val = step_scores.get(key)
            if val is not None:
                llm_scores[key].append(float(val))

        trajectory.append({
            "action": action,
            "action_idx": action_idx,
            "reward": reward,
            "done": done,
            "llm_scores": step_scores,
        })
        state = next_state

    return trajectory, total_reward, llm_scores


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseEvaluator(ABC):
    """All baseline implementations inherit from this class."""

    @abstractmethod
    def run(self, n_episodes: int) -> dict:
        """
        Evaluate for n_episodes and return aggregate metrics.

        Keys: mean_reward, std_reward, lcs, ess, hrs, comp
        (hrs maps to the judge's GRS — grounding risk score)
        """

    def _aggregate(self, rewards: list, score_lists: dict) -> dict:
        def _mean(lst):
            return float(np.mean(lst)) if lst else 0.0

        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "lcs": _mean(score_lists["LCS"]),
            "ess": _mean(score_lists["ESS"]),
            "hrs": _mean(score_lists["GRS"]),
            "comp": _mean(score_lists["COMP"]),
        }

    def _accumulate(self, all_scores: dict, ep_scores: dict) -> None:
        for key in all_scores:
            all_scores[key].extend(ep_scores[key])


# ---------------------------------------------------------------------------
# Block 4: Random and Majority baselines
# ---------------------------------------------------------------------------

class RandomBaseline(BaseEvaluator):
    """Uniformly random action selection over the full action space."""

    def __init__(self, eval_dataset: list):
        self.eval_dataset = eval_dataset

    def run(self, n_episodes: int) -> dict:
        env = ClaimEnv(self.eval_dataset)
        rewards = []
        all_scores = {"LCS": [], "ESS": [], "GRS": [], "COMP": []}

        for _ in range(n_episodes):
            _, total_reward, ep_scores = run_episode(
                env, lambda state: random.randint(0, N_ACTIONS - 1)
            )
            rewards.append(total_reward)
            self._accumulate(all_scores, ep_scores)

        return self._aggregate(rewards, all_scores)


class MajorityBaseline(BaseEvaluator):
    """Always selects the action most frequently taken during a random init pass."""

    def __init__(self, eval_dataset: list):
        self.eval_dataset = eval_dataset
        self._majority_idx = self._find_majority()

    def _find_majority(self) -> int:
        env = ClaimEnv(self.eval_dataset)
        counts = [0] * N_ACTIONS

        def action_fn(state):
            idx = random.randint(0, N_ACTIONS - 1)
            counts[idx] += 1
            return idx

        for _ in range(len(self.eval_dataset)):
            run_episode(env, action_fn)

        majority = int(np.argmax(counts))
        logger.info(
            "MajorityBaseline: majority action=%s (idx=%d)", ACTIONS[majority], majority
        )
        return majority

    def run(self, n_episodes: int) -> dict:
        env = ClaimEnv(self.eval_dataset)
        majority = self._majority_idx
        rewards = []
        all_scores = {"LCS": [], "ESS": [], "GRS": [], "COMP": []}

        for _ in range(n_episodes):
            _, total_reward, ep_scores = run_episode(env, lambda state: majority)
            rewards.append(total_reward)
            self._accumulate(all_scores, ep_scores)

        return self._aggregate(rewards, all_scores)


# ---------------------------------------------------------------------------
# Shared LLM prompt helpers (Blocks 5-7)
# ---------------------------------------------------------------------------

def _greedy_llm_prompt(state) -> str:
    return (
        "You are selecting the best next action for an evidence retrieval task.\n"
        f"Observation: {_state_summary(state)}\n"
        f"Available actions:\n{ACTION_DESCRIPTIONS}\n"
        "Respond with only the index number of the best action. No explanation."
    )


def _parse_action_idx(response: str, caller: str) -> int:
    """Extract an action index from an LLM response string, or fall back to random."""
    match = re.search(r"\d+", response.strip())
    if match:
        return int(np.clip(int(match.group()), 0, N_ACTIONS - 1))
    warnings.warn(f"[{caller}] LLM response could not be parsed, falling back to random.")
    return random.randint(0, N_ACTIONS - 1)


# ---------------------------------------------------------------------------
# Block 5: Greedy LLM baseline
# ---------------------------------------------------------------------------

class GreedyLLMBaseline(BaseEvaluator):
    """Single LLM call per step to select the predicted best action."""

    def __init__(self, eval_dataset: list, llm_client):
        self.eval_dataset = eval_dataset
        self.llm = llm_client

    def _action_fn(self, state) -> int:
        prompt = _greedy_llm_prompt(state)
        try:
            response, _ = self.llm.generate_structured(prompt)
            return _parse_action_idx(response, "GreedyLLMBaseline")
        except Exception as exc:
            warnings.warn(f"[GreedyLLMBaseline] LLM call failed ({exc}), falling back to random.")
            return random.randint(0, N_ACTIONS - 1)

    def run(self, n_episodes: int) -> dict:
        env = ClaimEnv(self.eval_dataset)
        rewards = []
        all_scores = {"LCS": [], "ESS": [], "GRS": [], "COMP": []}

        for _ in range(n_episodes):
            _, total_reward, ep_scores = run_episode(env, self._action_fn)
            rewards.append(total_reward)
            self._accumulate(all_scores, ep_scores)

        return self._aggregate(rewards, all_scores)


# ---------------------------------------------------------------------------
# Block 6: Few-shot LLM baseline
# ---------------------------------------------------------------------------

class FewShotLLMBaseline(BaseEvaluator):
    """LLM action selection augmented with few-shot examples from train_dataset."""

    def __init__(
        self,
        eval_dataset: list,
        train_dataset: list,
        llm_client,
        k: int = 3,
        selection_mode: str = "random",
    ):
        self.eval_dataset = eval_dataset
        self.train_dataset = train_dataset
        self.llm = llm_client
        self.k = k
        self.selection_mode = selection_mode

        self._examples = self._build_examples()

        self._train_embeddings = None
        if selection_mode == "similarity":
            self._train_embeddings = self._embed(
                [ex["observation"] for ex in self._examples]
            )

    def _build_examples(self) -> list:
        """One random-action pass over train_dataset to collect (observation, action) pairs."""
        env = ClaimEnv(self.train_dataset)
        examples = []

        def action_fn(state):
            idx = random.randint(0, N_ACTIONS - 1)
            examples.append({
                "observation": _state_summary(state),
                "action_idx": idx,
                "claim": state.claim,
            })
            return idx

        for _ in range(len(self.train_dataset)):
            run_episode(env, action_fn)

        logger.info("FewShotLLMBaseline: collected %d training examples.", len(examples))
        return examples

    def _embed(self, texts: list):
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            return model.encode(texts, normalize_embeddings=True)
        except ImportError:
            logger.warning(
                "sentence-transformers not available; similarity selection falls back to random."
            )
            return None

    def _select_examples(self, state, k: int) -> list:
        if self.selection_mode == "similarity" and self._train_embeddings is not None:
            query_emb = self._embed([_state_summary(state)])
            if query_emb is not None:
                sims = np.dot(self._train_embeddings, query_emb[0])
                top_k = np.argsort(sims)[::-1][:k]
                return [self._examples[i] for i in top_k]
        return random.sample(self._examples, min(k, len(self._examples)))

    def _action_fn(self, state, k: int) -> int:
        examples = self._select_examples(state, k)
        example_block = "\n".join(
            f"Observation: {ex['observation']} | Action chosen: {ex['action_idx']}"
            for ex in examples
        )
        prompt = (
            "Here are examples of good action selections:\n"
            f"{example_block}\n\n"
            "Now select the best action:\n"
            f"Observation: {_state_summary(state)}\n"
            f"Available actions:\n{ACTION_DESCRIPTIONS}\n"
            "Respond with only the index number."
        )
        try:
            response, _ = self.llm.generate_structured(prompt)
            return _parse_action_idx(response, "FewShotLLMBaseline")
        except Exception as exc:
            warnings.warn(f"[FewShotLLMBaseline] LLM call failed ({exc}), falling back to random.")
            return random.randint(0, N_ACTIONS - 1)

    def run(self, n_episodes: int, k: int = None) -> dict:
        k_shots = k if k is not None else self.k
        env = ClaimEnv(self.eval_dataset)
        rewards = []
        all_scores = {"LCS": [], "ESS": [], "GRS": [], "COMP": []}

        for _ in range(n_episodes):
            _, total_reward, ep_scores = run_episode(
                env, lambda state: self._action_fn(state, k_shots)
            )
            rewards.append(total_reward)
            self._accumulate(all_scores, ep_scores)

        return self._aggregate(rewards, all_scores)


# ---------------------------------------------------------------------------
# Block 7: Best-of-N baseline
# ---------------------------------------------------------------------------

class BestOfNBaseline(BaseEvaluator):
    """
    Inference-compute scaling: at each step, sample n LLM action suggestions,
    score each by immediate reward (no policy update), commit the best.
    """

    def __init__(self, eval_dataset: list, llm_client, n: int = 5):
        self.eval_dataset = eval_dataset
        self.llm = llm_client
        self.n = n
        self._current_env = None

    def _suggest_action(self, state) -> int:
        """One LLM suggestion using the greedy prompt from Block 5."""
        try:
            response, _ = self.llm.generate_structured(_greedy_llm_prompt(state))
            return _parse_action_idx(response, "BestOfNBaseline")
        except Exception:
            return random.randint(0, N_ACTIONS - 1)

    def _snapshot(self, env: ClaimEnv) -> dict:
        return {
            "state": copy.deepcopy(env.state),
            "prev_phi": getattr(env, "_prev_phi", 0.0),
            "last_judge_step": getattr(env, "_last_judge_step", 0),
        }

    def _restore(self, env: ClaimEnv, snap: dict) -> None:
        env.state = copy.deepcopy(snap["state"])
        env._prev_phi = snap["prev_phi"]
        env._last_judge_step = snap["last_judge_step"]

    def _action_fn(self, state) -> int:
        """Try n LLM suggestions; return the one with the highest immediate reward."""
        env = self._current_env
        candidates = [self._suggest_action(state) for _ in range(self.n)]

        snap = self._snapshot(env)
        best_idx = candidates[0]
        best_reward = float("-inf")

        for action_idx in candidates:
            self._restore(env, snap)
            action = ACTIONS[action_idx]
            payload = _build_payload(action, state)
            _, reward, _, _ = env.step(action, payload)
            if reward > best_reward:
                best_reward = reward
                best_idx = action_idx

        # restore pre-trial state so run_episode commits the winner cleanly
        self._restore(env, snap)
        return best_idx

    def run(self, n_episodes: int) -> dict:
        env = ClaimEnv(self.eval_dataset)
        self._current_env = env
        rewards = []
        all_scores = {"LCS": [], "ESS": [], "GRS": [], "COMP": []}
        total_llm_calls = 0

        for _ in range(n_episodes):
            traj, total_reward, ep_scores = run_episode(env, self._action_fn)
            rewards.append(total_reward)
            self._accumulate(all_scores, ep_scores)
            total_llm_calls += len(traj) * self.n

        avg_calls = total_llm_calls / max(n_episodes, 1)
        print(f"[BestOfNBaseline] avg LLM calls/episode: {avg_calls:.1f} (n={self.n} per step)")
        return self._aggregate(rewards, all_scores)


# ---------------------------------------------------------------------------
# Block 9: Imitation baseline (behavioral cloning at inference time)
# ---------------------------------------------------------------------------

class ImitationBaseline(BaseEvaluator):
    """
    Behavioral cloning at inference time: look up the closest recorded action
    for the current observation, keyed by claim then hashed observation.

    No gradient updates occur — this is pure lookup-based imitation.
    """

    def __init__(self, eval_dataset: list, trajectories_path: str, hamming_threshold: int = 3):
        self.eval_dataset = eval_dataset
        self.hamming_threshold = hamming_threshold
        # {claim -> list of (obs_hash_int, action_idx)}
        self._lookup: dict = {}
        self._load(trajectories_path)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _obs_hash(self, obs: str) -> int:
        return int(hashlib.md5(obs.encode()).hexdigest(), 16)

    def _hamming(self, a: int, b: int) -> int:
        return bin(a ^ b).count("1")

    def _load(self, path: str) -> None:
        n_trajectories = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                claim = rec.get("claim", "")
                entries = [
                    (self._obs_hash(step["obs"]), int(step["action"]))
                    for step in rec.get("steps", [])
                    if "obs" in step and "action" in step
                ]
                if claim not in self._lookup:
                    self._lookup[claim] = []
                self._lookup[claim].extend(entries)
                n_trajectories += 1

        n_claims = len(self._lookup)
        logger.info(
            "Imitation baseline loaded %d trajectories covering %d unique claims",
            n_trajectories,
            n_claims,
        )

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def _action_fn(self, state) -> int:
        obs = _state_summary(state)
        obs_hash = self._obs_hash(obs)
        claim = state.claim
        entries = self._lookup.get(claim, [])

        if not entries:
            logger.warning(
                "Imitation: no trajectories for claim='%.50s' obs='%.80s' — random fallback",
                claim,
                obs,
            )
            return random.randint(0, N_ACTIONS - 1)

        # Exact match
        for h, action_idx in entries:
            if h == obs_hash:
                return action_idx

        # Closest by Hamming distance over the 128-bit MD5 hash
        best_h, best_action = min(entries, key=lambda x: self._hamming(obs_hash, x[0]))
        dist = self._hamming(obs_hash, best_h)

        if dist <= self.hamming_threshold:
            return best_action

        logger.warning(
            "Imitation: no match within %d bits for claim='%.50s' obs='%.80s' "
            "(min dist=%d) — random fallback",
            self.hamming_threshold,
            claim,
            obs,
            dist,
        )
        return random.randint(0, N_ACTIONS - 1)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def run(self, n_episodes: int) -> dict:
        env = ClaimEnv(self.eval_dataset)
        rewards = []
        all_scores = {"LCS": [], "ESS": [], "GRS": [], "COMP": []}

        for _ in range(n_episodes):
            _, total_reward, ep_scores = run_episode(env, self._action_fn)
            rewards.append(total_reward)
            self._accumulate(all_scores, ep_scores)

        return self._aggregate(rewards, all_scores)
