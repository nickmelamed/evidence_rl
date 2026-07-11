import random

import numpy as np

from evid_rl_env.data.evidence_fetcher import fetch_evidence
from evid_rl_env.environment.actions import Actions
from evid_rl_env.environment.state import Evidence, State
from evid_rl_env.judge.evidence_labeler import EvidenceLabeler
from evid_rl_env.judge.llm_judge import LLMJudge
from evid_rl_env.judge.reward import RewardFunction

# JudgeLLMClient and LLMJudge are shared across all ClaimEnv instances.
# LLMJudge now uses a sqlite3 cache (crash-safe, no POSIX semaphores).
_judge_llm_cache: dict = {}       # model_name -> JudgeLLMClient
_llm_judge_cache: dict = {}       # model_name -> LLMJudge
_evidence_labeler_cache: dict = {}  # model_name -> EvidenceLabeler


def _get_llm_judge(model_name: str, seed: int) -> LLMJudge:
    from evid_rl_env.agent.llm_client import JudgeLLMClient
    if model_name not in _judge_llm_cache:
        _judge_llm_cache[model_name] = JudgeLLMClient(model_name=model_name, seed=seed)
    # seed is fixed at init; the cached client is reused across all ClaimEnv instances
    if model_name not in _llm_judge_cache:
        _llm_judge_cache[model_name] = LLMJudge(_judge_llm_cache[model_name])
    return _llm_judge_cache[model_name]


def _get_evidence_labeler(model_name: str, seed: int) -> EvidenceLabeler:
    # Reuses the same cached JudgeLLMClient as _get_llm_judge (same default
    # model, Qwen/Qwen2.5-1.5B-Instruct) — labeling evidence doesn't need a
    # separate model download, just a second cached-by-content-hash wrapper
    # around the same client.
    from evid_rl_env.agent.llm_client import JudgeLLMClient
    if model_name not in _judge_llm_cache:
        _judge_llm_cache[model_name] = JudgeLLMClient(model_name=model_name, seed=seed)
    if model_name not in _evidence_labeler_cache:
        _evidence_labeler_cache[model_name] = EvidenceLabeler(_judge_llm_cache[model_name])
    return _evidence_labeler_cache[model_name]


def _default_embedder(text: str):
    # Lazy import so environment.py never gains a module-level sentence-
    # transformers/torch dependency — the cost is only paid the first time
    # RERANK actually runs, not at ClaimEnv construction or import time.
    from evid_rl_env.agent.state_encoder import embed_text
    return embed_text(text)


class ClaimEnv:
    def __init__(self, dataset, judge_model=None, seed: int = 42, llm_judge=None,
                 embedder=None, evidence_labeler=None):
        self.dataset = dataset
        self.state = None
        self.current_sample = None
        self.reward_fn = RewardFunction()

        judge_model_name = judge_model or "Qwen/Qwen2.5-1.5B-Instruct"

        if llm_judge is not None:
            # Test/injection hook: bypass the global model-backed cache entirely
            # so callers (e.g. tests) can supply a lightweight mock judge without
            # ever importing transformers/torch via agent.llm_client.
            self.llm_judge = llm_judge
        else:
            self.llm_judge = _get_llm_judge(judge_model_name, seed)
        self._last_judge_step = -1

        # Test/injection hook for RERANK's claim-similarity sort — same shape
        # as llm_judge= above. embedder(text) must return an embedding vector
        # or None (treated as "unavailable", falls back to a text-length sort).
        self.embedder = embedder if embedder is not None else _default_embedder

        # Test/injection hook for evidence stance/reliability labeling — same
        # shape again. evidence_labeler.label(claim, text) must return one of
        # support/contradict/neutral/adversarial.
        if evidence_labeler is not None:
            self.evidence_labeler = evidence_labeler
        else:
            self.evidence_labeler = _get_evidence_labeler(judge_model_name, seed)

    def _evidence_diversity_bonus(self, new_evidence_id):
        """Returns a small bonus if this evidence id hasn't been selected before."""
        return 0.05 if new_evidence_id not in self.state.selected_evidence_ids else 0.0

    @staticmethod
    def _claim_similarity(claim: str, text: str) -> float:
        """Word-overlap proxy: fraction of >3-char claim words that appear in text."""
        claim_words = {w.lower().strip(".,!?;:\"'()[]") for w in claim.split() if len(w) > 3}
        if not claim_words:
            return 0.0
        text_lower = text.lower()
        return sum(1 for w in claim_words if w in text_lower) / len(claim_words)

    def reset(self):
        self._prev_phi = 0.0
        # -1 (not 0) so the very first debate action at steps_taken==1 clears
        # the "every 2 steps" gate (1 - (-1) >= 2) instead of silently reusing
        # the initial 0.0 score.
        self._last_judge_step = -1
        self.current_sample = random.choice(self.dataset)
        claim = self.current_sample["claim"]
        search_query = self.current_sample.get("search_query", claim)

        try:
            raw_evidence = fetch_evidence(claim, search_query)
            evidence_pool = [
                Evidence(id=i, text=e["content"], label=self.evidence_labeler.label(claim, e["content"]))
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
        task_success = None
        done = False

        # action handling
        if action == Actions.SELECT:
            doc = next((e for e in s.evidence_pool if e.id == payload), None)
            if doc and doc not in s.selected_evidence:
                s.selected_evidence.append(doc)
                diversity = self._evidence_diversity_bonus(payload)
                s.selected_evidence_ids.add(payload)
                sim = self._claim_similarity(s.claim, doc.text)
                reward = 0.1 + diversity + 0.1 * sim
            elif doc is not None:
                # doc is in the pool but already selected — penalise redundancy
                reward = -0.1

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
                reward = -1.0
                done = True

            # penalty for not taking enough steps (episode continues — this
            # only blocks *premature* finalization, it doesn't end the episode)
            elif s.steps_taken <= 2:
                reward = -0.5

            else:
                # use the agent's own ASSIGN_CONFIDENCE value if it called that
                # action this episode; otherwise fall back to the evidence-count
                # heuristic so FINALIZE still works without it.
                confidence = (
                    s.confidence if s.confidence is not None
                    else min(1.0, len(s.selected_evidence) / 3)
                )

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

                # llm reward computed first so base_reward's documented 0.10
                # weight is actually applied instead of being silently
                # redistributed into F1/CA (RewardFunction.compute's None-path)
                llm_reward, llm_scores = self.llm_judge.compute_reward(
                    claim=s.claim,
                    reasoning=reasoning,
                    evidence=s.selected_evidence
                )

                # hybrid reward calculation
                base_reward = self.reward_fn.compute(s, final_output, llm_reward=llm_reward)

                # RLHF
                alpha = 0.3
                reward = (1 - alpha) * base_reward + alpha * llm_reward

                # penalty for empty debate
                if not s.debate_history:
                    reward -= 0.3

                # reward for evidence use
                reward += 0.2 * len(s.selected_evidence)

                # bounded, single-purpose calibration signal for the curriculum
                # (distinct from `reward`, which mixes in shaping/evidence terms)
                task_success = max(0.0, min(1.0, 1.0 - abs(confidence - true_score)))

                done = True

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
                                label=self.evidence_labeler.label(s.claim, e["content"])
                            )
                            for i, e in enumerate(raw)
                        ]
                        s.evidence_pool.extend(new_evidence)
                        reward = 0.05 * len(new_evidence)
                    except Exception:
                        reward = 0.0

        elif action == Actions.RERANK:
            # Reorder selected_evidence by descending cosine similarity to the
            # claim. Falls back to a text-length sort if the embedder is
            # unavailable (sentence-transformers not installed, or an injected
            # test embedder returns None) — never a hard failure.
            if s.selected_evidence:
                claim_emb = self.embedder(s.claim)
                if claim_emb is not None:
                    def _claim_sim(e, _claim_emb=claim_emb):
                        emb = self.embedder(e.text)
                        return float(np.dot(_claim_emb, emb)) if emb is not None else 0.0
                    s.selected_evidence.sort(key=_claim_sim, reverse=True)
                else:
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

        elif action == Actions.ASSIGN_CONFIDENCE:
            if payload is not None:
                try:
                    s.confidence = max(0.0, min(1.0, float(payload)))
                    reward = 0.05
                except (TypeError, ValueError):
                    reward = 0.0

        elif action == Actions.CHALLENGE_EVIDENCE:
            if payload and isinstance(payload, dict) and payload.get("argument"):
                target_id = payload.get("evidence_id")
                target = next((e for e in s.evidence_pool if e.id == target_id), None)
                if target is not None:
                    s.challenged_evidence_ids.add(target_id)
                    s.debate_history.append(f"CHALLENGE({target_id}): " + str(payload["argument"]))

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

                    # bonus for correctly targeting adversarial evidence, penalty
                    # for targeting evidence that actually supports the claim
                    if target.label == "adversarial":
                        reward += 0.1
                    elif target.label == "support":
                        reward -= 0.05

        elif action == Actions.REQUEST_CLARIFICATION:
            if s.query_count >= s.max_queries:
                reward = -0.1  # shares QUERY's per-episode budget
            else:
                s.query_count += 1
                if payload and isinstance(payload, str):
                    s.debate_history.append("CLARIFY: " + payload)
                    try:
                        from evid_rl_env.data.evidence_fetcher import fetch_evidence
                        raw = fetch_evidence(s.claim, payload)
                        new_evidence = [
                            Evidence(
                                id=len(s.evidence_pool) + i,
                                text=e["content"],
                                label=self.evidence_labeler.label(s.claim, e["content"])
                            )
                            for i, e in enumerate(raw)
                        ]
                        s.evidence_pool.extend(new_evidence)
                        reward = 0.05 * len(new_evidence)
                    except Exception:
                        reward = 0.0

        elif action == Actions.HEDGE:
            if payload and isinstance(payload, dict) and payload.get("argument"):
                s.debate_history.append("HEDGE: " + str(payload["argument"]))

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

        # step limit termination (does not override a FINALIZE that already ended the episode)
        if not done and s.is_done():
            # penalize not finalizing
            reward = -0.2
            done = True

        # fallback
        if reward == 0.0:
            if action == Actions.REMOVE:
                reward = -0.05

        # Potential-based shaping: F(s,s') = gamma * Phi(s') - Phi(s)
        # Phi = current LLM judge score (higher = better state). Terminal states
        # use Phi(terminal) = 0 by convention so shaping telescopes to -Phi(s0)
        # across the episode (Ng et al. 1999 policy-invariance requirement)
        # instead of silently vanishing on FINALIZE/step-limit transitions.
        phi_next = 0.0 if done else s.last_llm_score
        phi_prev = getattr(self, "_prev_phi", 0.0)
        self._prev_phi = phi_next
        shaping = 0.99 * phi_next - phi_prev
        reward += 0.1 * shaping

        return s, reward, done, {
            "llm_scores": llm_scores,
            "llm_reward": llm_reward,
            "task_success": task_success,
        }
