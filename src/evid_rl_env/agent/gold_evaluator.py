import numpy as np


class GoldEvaluator:
    """Runs a held-out eval split and re-scores each episode's final reasoning
    with one or more gold judges that never participate in training reward,
    to check the training judge's scores against an independent signal.

    Structurally a sibling of agent/evaluator.py's Evaluator, but instead of
    aggregating the training judge's own running scores, it captures the
    training judge's FINALIZE-step scores (already computed by
    ClaimEnv.step and returned in info["llm_scores"] / info["llm_reward"] —
    no extra training-judge calls needed) and compares them to fresh
    gold-judge scores on the exact same (claim, reasoning, evidence) triple.

    Unlike Evaluator, this rollout samples stochastically (greedy=False)
    rather than greedy/argmax. Evaluator's job is measuring the policy's
    true performance, where greedy is the right choice; GoldEvaluator's job
    is auditing the reward signal itself, which needs episodes that reach
    FINALIZE to produce anything to compare — an under-trained policy's
    greedy/argmax action can get stuck never selecting FINALIZE within the
    step budget (no exploration to escape it), starving gold_eval of any
    scored episodes early in training. Stochastic sampling avoids that.
    """

    _DIMENSIONS = ("LCS", "ESS", "GRS", "COMP", "BIAS")

    def __init__(self, env, policy, gold_judges: list, n_episodes: int = 20):
        self.env = env
        self.policy = policy
        self.gold_judges = gold_judges
        self.n_episodes = n_episodes

    def evaluate(self) -> dict:
        proxy_dim_scores = {k: [] for k in self._DIMENSIONS}
        gold_dim_scores = {k: [] for k in self._DIMENSIONS}
        proxy_rewards = []
        gold_rewards = []
        accuracies = []
        hard_correct = []
        escalated_flags = []

        for _ in range(self.n_episodes):
            if hasattr(self.policy, "reset_episode_cache"):
                self.policy.reset_episode_cache()
            state = self.env.reset()
            done = False
            info = {}
            while not done:
                action, payload, _ = self.policy.act(state, greedy=False)
                state, _reward, done, info = self.env.step(action, payload)

            # task_success is only set on the FINALIZE step (see
            # environment.py) — None means the episode was cut off by the
            # step/token limit before reaching a judged final answer.
            proxy = info.get("llm_scores")
            if info.get("task_success") is None or not proxy:
                continue

            reasoning = " ".join(state.debate_history)
            evidence = state.selected_evidence
            true_label = float(self.env.current_sample.get("label", 0.5))
            # mirrors ClaimEnv.step's FINALIZE confidence fallback so the
            # accuracy signal here matches what the training reward saw
            confidence = (
                state.confidence if state.confidence is not None
                else min(1.0, len(state.selected_evidence) / 3)
            )

            per_judge_rewards = []
            per_judge_dims = {k: [] for k in self._DIMENSIONS}
            for judge in self.gold_judges:
                reward, scores = judge.compute_reward(state.claim, reasoning, evidence)
                per_judge_rewards.append(reward)
                for k in self._DIMENSIONS:
                    if k in scores:
                        per_judge_dims[k].append(float(scores[k]))

            proxy_rewards.append(float(info.get("llm_reward", 0.0)))
            gold_rewards.append(float(np.mean(per_judge_rewards)))
            for k in self._DIMENSIONS:
                if k in proxy:
                    proxy_dim_scores[k].append(float(proxy[k]))
                if per_judge_dims[k]:
                    gold_dim_scores[k].append(float(np.mean(per_judge_dims[k])))

            accuracies.append(1.0 - abs(confidence - true_label))
            hard_correct.append(1.0 if (confidence >= 0.5) == (true_label >= 0.5) else 0.0)
            # only architectures with an escalation tier (EscalatingJudge)
            # set this key at all — its absence, not False, is what means
            # "not applicable" (see result["escalation_rate"] below)
            if "escalated" in proxy:
                escalated_flags.append(bool(proxy["escalated"]))

        result = {
            "n_episodes": self.n_episodes,
            "n_scored": len(proxy_rewards),
            "proxy_reward_mean": float(np.mean(proxy_rewards)) if proxy_rewards else None,
            "gold_reward_mean": float(np.mean(gold_rewards)) if gold_rewards else None,
            "proxy_gold_correlation": self._safe_corr(proxy_rewards, gold_rewards),
            "outcome_accuracy": float(np.mean(accuracies)) if accuracies else None,
            "outcome_accuracy_hard": float(np.mean(hard_correct)) if hard_correct else None,
            # None (not 0.0) means "this judge architecture doesn't escalate
            # at all", distinct from "it escalated 0% of scored episodes"
            "escalation_rate": float(np.mean(escalated_flags)) if escalated_flags else None,
            "dimensions": {},
        }
        for k in self._DIMENSIONS:
            p, g = proxy_dim_scores[k], gold_dim_scores[k]
            if not p or not g:
                continue
            result["dimensions"][k] = {
                "proxy_mean": float(np.mean(p)),
                "gold_mean": float(np.mean(g)),
                "mean_abs_disagreement": float(np.mean([abs(pi - gi) for pi, gi in zip(p, g)])),
            }
        return result

    @staticmethod
    def _safe_corr(a: list, b: list):
        if len(a) < 2 or len(b) < 2:
            return None
        a_arr, b_arr = np.array(a), np.array(b)
        if np.std(a_arr) == 0 or np.std(b_arr) == 0:
            return None
        return float(np.corrcoef(a_arr, b_arr)[0, 1])
