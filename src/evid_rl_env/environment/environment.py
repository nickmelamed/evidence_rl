import random
import numpy as np

from evid_rl_env.environment.state import State, Evidence
from evid_rl_env.environment.actions import Actions
from evid_rl_env.judge.reward import RewardFunction
from evid_rl_env.judge.llm_judge import LLMJudge
from evid_rl_env.agent.llm_client import LLMClient
from evid_rl_env.data.evidence_fetcher import fetch_evidence


class ClaimEnv:
    def __init__(self, dataset):
        self.dataset = dataset
        self.state = None
        self.current_sample = None
        self.reward_fn = RewardFunction()

        llm = LLMClient()
        self.llm_judge = LLMJudge(llm, weight=0.5)

    def reset(self):
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

        # action handling
        if action == Actions.SELECT:
            doc = next((e for e in s.evidence_pool if e.id == payload), None)
            if doc and doc not in s.selected_evidence:
                s.selected_evidence.append(doc)

        elif action == Actions.REMOVE:
            s.selected_evidence = [
                e for e in s.selected_evidence if e.id != payload
            ]

        elif action == Actions.SUPPORT:
            if payload:
                s.debate_history.append("SUPPORT: " + str(payload))

                # mid-episode shaping
                partial_reasoning = " ".join(s.debate_history)

                llm_reward, _ = self.llm_judge.compute_reward(
                    claim=s.claim,
                    reasoning=partial_reasoning,
                    evidence=s.selected_evidence
                )

                reward = 0.1 + 0.1 * llm_reward  # base + LLM shaping


        elif action == Actions.CONTRADICT:
            if payload:
                s.debate_history.append("CONTRADICT: " + str(payload))

                partial_reasoning = " ".join(s.debate_history)

                llm_reward, _ = self.llm_judge.compute_reward(
                    claim=s.claim,
                    reasoning=partial_reasoning,
                    evidence=s.selected_evidence
                )

                reward = 0.1 + 0.1 * llm_reward


        elif action == Actions.FINALIZE:
            # build the final output 
            reasoning = " ".join(s.debate_history)

            # penalty for 0 evidence 
            if len(s.selected_evidence) == 0:
                return s, -1.0, True, {}
            
            # penalty for not taking enough steps
            if s.steps_taken <= 2:
                return s, -0.5, False, {}

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

            return s, reward, True, {}

        # step limit termination
        if s.is_done():
            # penalize not finalizing
            return s, -0.2, True, {}

        # fallback
        if reward == 0.0:
            if action == Actions.SELECT:
                reward = 0.1
            elif action == Actions.REMOVE:
                reward = -0.05  # discourage useless removal
        

        return s, reward, False, {}
