import atexit
import random
import numpy as np

from evid_rl_env.environment.state import State, Evidence
from evid_rl_env.environment.actions import Actions
from evid_rl_env.judge.reward import RewardFunction
from evid_rl_env.judge.llm_judge import LLMJudge
from evid_rl_env.data.evidence_fetcher import fetch_evidence

# Both the JudgeLLMClient and the LLMJudge (which holds a shelve file handle)
# are shared across all ClaimEnv instances.  Opening the same shelve file from
# multiple LLMJudge objects simultaneously causes gdbm to create a POSIX
# semaphore for locking; if any handle is not closed at exit that semaphore
# leaks and can cause a segfault during interpreter shutdown cleanup.
_judge_llm_cache: dict = {}   # model_name -> JudgeLLMClient
_llm_judge_cache: dict = {}   # model_name -> LLMJudge  (owns the shelve handle)


def _get_llm_judge(model_name: str, seed: int) -> LLMJudge:
    from evid_rl_env.agent.llm_client import JudgeLLMClient
    if model_name not in _judge_llm_cache:
        _judge_llm_cache[model_name] = JudgeLLMClient(model_name=model_name, seed=seed)
    else:
        _judge_llm_cache[model_name].seed = seed

    if model_name not in _llm_judge_cache:
        _llm_judge_cache[model_name] = LLMJudge(_judge_llm_cache[model_name], weight=0.5)
    return _llm_judge_cache[model_name]


def _close_llm_judges():
    """atexit handler — flush and close every shared shelve handle before exit."""
    for judge in _llm_judge_cache.values():
        try:
            judge.close()
        except Exception:
            pass


atexit.register(_close_llm_judges)


class ClaimEnv:
    def __init__(self, dataset, judge_model=None, seed: int = 42):
        self.dataset = dataset
        self.state = None
        self.current_sample = None
        self.reward_fn = RewardFunction()

        judge_model_name = judge_model or "Qwen/Qwen2.5-1.5B-Instruct"
        self.llm_judge = _get_llm_judge(judge_model_name, seed)
        self._last_judge_step = 0

    def _evidence_diversity_bonus(self, new_evidence_id):
        """Returns a small bonus if this evidence id hasn't been selected before."""
        return 0.05 if new_evidence_id not in self.state.selected_evidence_ids else 0.0

    def reset(self):
        self._prev_phi = 0.0
        self._last_judge_step = 0
        self.current_sample = random.choice(self.dataset)
        claim = self.current_sample["claim"]
        search_query = self.current_sample.get("search_query", claim)

        try:
            raw_evidence = fetch_evidence(claim, search_query)
            evidence_pool = [
                Evidence(id=i, text=e["content"], label=e.get("label", "neutral"))
                for i, e in enumerate(raw_evidence)
            ]
        except Exception as exc:
            print(f"[WARNING] Tavily fetch failed, falling back to seed_claims.json data: {exc}")
            evidence_pool = []

        if not evidence_pool and "evidence" in self.current_sample:
            evidence_pool = [
                Evidence(id=i, text=e["text"], label=e.get("label", "neutral"))
                for i, e in enumerate(self.current_sample["evidence"])
            ]

        self.state = State(claim=claim, evidence_pool=evidence_pool)
        return self.state

    def step(self, action, payload):
        s = self.state
        s.steps_taken += 1
        reward = 0.0
        llm_reward = 0.0
        llm_scores = {}

        # action handling
        if action == Actions.SELECT:
            doc = next((e for e in s.evidence_pool if e.id == payload), None)
            if doc and doc not in s.selected_evidence:
                s.selected_evidence.append(doc)
                diversity = self._evidence_diversity_bonus(payload)
                s.selected_evidence_ids.add(payload)
                reward = 0.1 + diversity

        elif action == Actions.REMOVE:
            s.selected_evidence = [
                e for e in s.selected_evidence if e.id != payload
            ]

        elif action == Actions.SUPPORT:
            if payload:
                s.debate_history.append("SUPPORT: " + str(payload))

                partial_reasoning = " ".join(s.debate_history)
                if s.steps_taken - self._last_judge_step >= 2:
                    llm_reward, llm_scores = self.llm_judge.compute_reward(
                        claim=s.claim,
                        reasoning=partial_reasoning,
                        evidence=s.selected_evidence
                    )
                    self._last_judge_step = s.steps_taken
                else:
                    llm_reward, llm_scores = s.last_llm_score, {}
                delta = llm_reward - s.last_llm_score
                s.last_llm_score = llm_reward
                reward = 0.05 + 0.15 * delta


        elif action == Actions.CONTRADICT:
            if payload:
                s.debate_history.append("CONTRADICT: " + str(payload))

                partial_reasoning = " ".join(s.debate_history)
                if s.steps_taken - self._last_judge_step >= 2:
                    llm_reward, llm_scores = self.llm_judge.compute_reward(
                        claim=s.claim,
                        reasoning=partial_reasoning,
                        evidence=s.selected_evidence
                    )
                    self._last_judge_step = s.steps_taken
                else:
                    llm_reward, llm_scores = s.last_llm_score, {}
                delta = llm_reward - s.last_llm_score
                s.last_llm_score = llm_reward
                reward = 0.05 + 0.15 * delta


        elif action == Actions.FINALIZE:
            # build the final output 
            reasoning = " ".join(s.debate_history)

            # penalty for 0 evidence
            if len(s.selected_evidence) == 0:
                return s, -1.0, True, {"llm_scores": llm_scores, "llm_reward": llm_reward}

            # penalty for not taking enough steps
            if s.steps_taken <= 2:
                return s, -0.5, False, {"llm_scores": llm_scores, "llm_reward": llm_reward}

            # confidence heuristic
            confidence = min(1.0, len(s.selected_evidence) / 3)

            # use dataset ground truth if available
            if "label" in self.current_sample:
                true_score = float(self.current_sample["label"])
            else:
                # fallback heuristic
                true_score = (
                    np.mean([
                        1 if e.label == "support" else 0
                        for e in s.selected_evidence
                    ])
                    if s.selected_evidence else 0
                )

            final_output = {
                "reasoning": reasoning,
                "confidence": confidence,
                "true_score": true_score
            }

            # hybrid reward calculation 
            # base reward
            base_reward = self.reward_fn.compute(s, final_output)

            # llm reward
            llm_reward, llm_scores = self.llm_judge.compute_reward(
                claim=s.claim,
                reasoning=reasoning,
                evidence=s.selected_evidence
            )

            # RLHF
            alpha = 0.3
            reward = (1 - alpha) * base_reward + alpha * llm_reward

            # penalty for empty debate 
            if not s.debate_history:
                reward -= 0.3

            # reward for evidence use
            reward += 0.2 * len(s.selected_evidence)

            return s, reward, True, {"llm_scores": llm_scores, "llm_reward": llm_reward}

        elif action == Actions.QUERY:
            if s.query_count >= s.max_queries:
                reward = -0.1  # penalise exceeding query budget
            else:
                s.query_count += 1
                if payload and isinstance(payload, str):
                    try:
                        from evid_rl_env.data.evidence_fetcher import fetch_evidence
                        raw = fetch_evidence(s.claim, payload)
                        new_evidence = [
                            Evidence(
                                id=len(s.evidence_pool) + i,
                                text=e["content"],
                                label=e.get("label", "neutral")
                            )
                            for i, e in enumerate(raw)
                        ]
                        s.evidence_pool.extend(new_evidence)
                        reward = 0.05 * len(new_evidence)
                    except Exception:
                        reward = 0.0

        elif action == Actions.RERANK:
            # Reorder selected_evidence by descending text length as a simple
            # relevance heuristic; swap for embedding similarity once encoder is wired in
            if s.selected_evidence:
                s.selected_evidence.sort(key=lambda e: len(e.text), reverse=True)
            reward = 0.02

        elif action == Actions.SUMMARIZE:
            if s.selected_evidence and payload and isinstance(payload, dict):
                summary_text = payload.get("summary", "")
                if summary_text:
                    s.summary = summary_text
                    s.debate_history.append("SUMMARY: " + summary_text)
                    reward = 0.05
                else:
                    reward = 0.0
            else:
                reward = 0.0

        elif action == Actions.CONCEDE:
            if payload and isinstance(payload, dict):
                concession = payload.get("argument", "")
                if concession:
                    s.debate_history.append("CONCEDE: " + concession)
                    partial_reasoning = " ".join(s.debate_history)
                    if s.steps_taken - self._last_judge_step >= 2:
                        llm_reward, llm_scores = self.llm_judge.compute_reward(
                            claim=s.claim,
                            reasoning=partial_reasoning,
                            evidence=s.selected_evidence
                        )
                        self._last_judge_step = s.steps_taken
                    else:
                        llm_reward, llm_scores = s.last_llm_score, {}
                    delta = llm_reward - s.last_llm_score
                    s.last_llm_score = llm_reward
                    reward = 0.05 + 0.1 * delta

        # step limit termination
        if s.is_done():
            # penalize not finalizing
            return s, -0.2, True, {"llm_scores": llm_scores, "llm_reward": llm_reward}

        # fallback
        if reward == 0.0:
            if action == Actions.REMOVE:
                reward = -0.05

        # Potential-based shaping: F(s,s') = gamma * Phi(s') - Phi(s)
        # Phi = current LLM judge score (higher = better state)
        phi_next = s.last_llm_score
        phi_prev = getattr(self, "_prev_phi", 0.0)
        self._prev_phi = phi_next
        shaping = 0.99 * phi_next - phi_prev
        reward += 0.1 * shaping

        return s, reward, False, {"llm_scores": llm_scores, "llm_reward": llm_reward}
