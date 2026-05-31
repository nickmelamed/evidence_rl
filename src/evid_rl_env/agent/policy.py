import numpy as np
from evid_rl_env.environment.actions import Actions, ACTIONS
from evid_rl_env.agent.llm_client import LLMClient
from evid_rl_env.agent.state_encoder import encode_state_semantic, get_state_dim, SEMANTIC_AVAILABLE

STATE_DIM = get_state_dim()

def encode_state(state):
    return encode_state_semantic(state)

class ActorCriticPolicy:
    def __init__(self, n_actions, state_dim=None, model_name=None, seed: int = 42):
        if state_dim is None: state_dim = get_state_dim()
        self.actions = ACTIONS
        self.n_actions = n_actions
        self.state_dim = state_dim

        # Actor
        self.actor_params = np.random.randn(state_dim, n_actions) * 0.01

        # Critic (value function)
        self.value_params = np.zeros(state_dim)

        from evid_rl_env.agent.llm_client import LLMClient
        actor_model = model_name or "google/gemma-2-2b-it"
        # Propagate seed so LLMClient calls transformers.set_seed once at init.
        self.llm = LLMClient(model_name=actor_model, seed=seed)

    def reset_episode_cache(self):
        self._arg_cache = {}

    def get_logits(self, state):
        features = encode_state(state)
        return features @ self.actor_params

    def get_probs(self, state):
        logits = self.get_logits(state)
        logits -= np.max(logits)
        exp = np.exp(logits)
        return exp / np.sum(exp)

    def get_value(self, state):
        features = encode_state(state)
        return features @ self.value_params

    def act(self, state, greedy: bool = False, force_action_idx: int = None):
        if not hasattr(self, "_arg_cache"):
            self._arg_cache = {}

        probs = self.get_probs(state)  # fresh array — safe to mutate in place

        # Mask 1: QUERY when per-episode budget exhausted
        if state.query_count >= state.max_queries:
            probs[self.actions.index(Actions.QUERY)] = 0
            probs /= probs.sum()

        # Mask 2: argument-generating and evidence-removing actions require at least
        # one piece of evidence to be selected first.  Without evidence they call the
        # LLM with an empty context, produce no useful signal, and (critically) fill
        # every episode step with expensive LLM calls — the PPO-collapse OOM pattern.
        # Masking before sampling keeps the gradient attribution correct.
        if not state.selected_evidence and state.evidence_pool:
            for _a in (Actions.SUPPORT, Actions.CONTRADICT,
                       Actions.CONCEDE, Actions.SUMMARIZE, Actions.REMOVE):
                probs[self.actions.index(_a)] = 0
            _s = probs.sum()
            probs /= _s if _s > 0 else 1.0

        self.last_entropy = float(-np.sum(probs * np.log(probs + 1e-8)))
        self.last_probs = probs.copy()

        if force_action_idx is not None:
            idx = force_action_idx
            action = self.actions[idx]
            # Redirect invalid forced actions (e.g. from bandit) to SELECT
            if (action in (Actions.SUPPORT, Actions.CONTRADICT,
                           Actions.CONCEDE, Actions.SUMMARIZE, Actions.REMOVE)
                    and not state.selected_evidence and state.evidence_pool):
                action = Actions.SELECT
                idx = self.actions.index(Actions.SELECT)
        else:
            idx = int(np.argmax(probs)) if greedy else np.random.choice(self.n_actions, p=probs)
            action = self.actions[idx]

        if action == Actions.SELECT:
            doc = np.random.choice(state.evidence_pool)
            return action, doc.id, idx

        elif action in [Actions.SUPPORT, Actions.CONTRADICT]:
            texts = [e.text for e in state.selected_evidence]
            cache_key = (str(action), frozenset(e.id for e in state.selected_evidence))
            if cache_key in self._arg_cache:
                argument, tokens = self._arg_cache[cache_key]
            else:
                argument, tokens = self.llm.generate(
                    f"""
Claim: {state.claim}
Evidence: {texts}
Write a concise {action.lower()} argument.
"""
                )
                self._arg_cache[cache_key] = (argument, tokens)

            return action, {
                "argument": argument,
                "evidence_ids": [e.id for e in state.selected_evidence],
                "tokens": tokens
            }, idx

        elif action == Actions.QUERY:
            query_prompt = f"Given claim: {state.claim}\nWrite a short search query to find more relevant evidence."
            query_text, tokens = self.llm.generate(query_prompt)
            return action, query_text.strip(), idx

        elif action == Actions.RERANK:
            return action, None, idx

        elif action == Actions.SUMMARIZE:
            if state.selected_evidence:
                texts = [e.text for e in state.selected_evidence]
                cache_key = (str(action), frozenset(e.id for e in state.selected_evidence))
                if cache_key in self._arg_cache:
                    summary_text, tokens = self._arg_cache[cache_key]
                else:
                    summary_prompt = f"Claim: {state.claim}\nEvidence:\n{texts}\nWrite a one-sentence summary of how this evidence relates to the claim."
                    summary_text, tokens = self.llm.generate(summary_prompt)
                    self._arg_cache[cache_key] = (summary_text, tokens)
                return action, {"summary": summary_text.strip(), "tokens": tokens}, idx
            return action, {"summary": "", "tokens": 0}, idx

        elif action == Actions.CONCEDE:
            texts = [e.text for e in state.selected_evidence]
            cache_key = (str(action), frozenset(e.id for e in state.selected_evidence))
            if cache_key in self._arg_cache:
                argument, tokens = self._arg_cache[cache_key]
            else:
                concede_prompt = f"Claim: {state.claim}\nEvidence: {texts}\nWrite a concise acknowledgement of the strongest counterpoint to the claim."
                argument, tokens = self.llm.generate(concede_prompt)
                self._arg_cache[cache_key] = (argument, tokens)
            return action, {"argument": argument, "evidence_ids": [e.id for e in state.selected_evidence], "tokens": tokens}, idx

        return action, None, idx

    def grad_log_prob(self, state, action_idx, features=None, probs=None):
        if features is None:
            features = encode_state(state)
        if probs is None:
            logits = features @ self.actor_params
            logits -= np.max(logits)
            exp = np.exp(logits)
            probs = exp / np.sum(exp)
        grad = -probs.copy()
        grad[action_idx] += 1
        return np.outer(features, grad)

    def save(self, path: str) -> None:
        """Persist learnable parameters to a .npz checkpoint.

        Saves actor_params, value_params, and enough metadata to reconstruct
        the policy without the original config.  The LLM pipeline is not
        serialised — it is re-instantiated from model_name on load.
        """
        np.savez_compressed(
            path,
            actor_params=self.actor_params,
            value_params=self.value_params,
            n_actions=np.array([self.n_actions]),
            state_dim=np.array([self.state_dim]),
            model_name=np.array([self.llm.model_name], dtype="U256"),
        )

    @classmethod
    def load(cls, path: str, seed: int = 42) -> "ActorCriticPolicy":
        """Reconstruct a policy from a .npz checkpoint produced by save().

        The LLM pipeline is re-initialised from the stored model_name, so the
        first call may trigger a model download if not already cached.
        """
        data = np.load(path, allow_pickle=False)
        n_actions  = int(data["n_actions"][0])
        state_dim  = int(data["state_dim"][0])
        model_name = str(data["model_name"][0])
        policy = cls(n_actions=n_actions, state_dim=state_dim, model_name=model_name, seed=seed)
        policy.actor_params = data["actor_params"]
        policy.value_params = data["value_params"]
        return policy


class BanditPolicyWrapper:
    """Adapts a LinUCBBandit for use with Evaluator.

    Mirrors the training loop: bandit selects the action, the inner
    ActorCriticPolicy generates the payload (LLM arguments, queries, etc.).
    """

    def __init__(self, bandit, inner_policy: ActorCriticPolicy):
        self.bandit = bandit
        self._inner = inner_policy
        self.actions = inner_policy.actions
        self.n_actions = inner_policy.n_actions
        self.state_dim = inner_policy.state_dim
        self.llm = getattr(inner_policy, "llm", None)
        self.last_entropy = 0.0
        self.last_probs = np.zeros(inner_policy.n_actions)

    def act(self, state, greedy: bool = True):
        x = encode_state(state)
        action_idx = self.bandit.select_action(x)
        action = self.actions[action_idx]
        _, payload, _ = self._inner.act(state, greedy=greedy, force_action_idx=action_idx)
        self.last_entropy = self._inner.last_entropy
        self.last_probs = self._inner.last_probs
        return action, payload, action_idx

    def reset_episode_cache(self):
        self._inner.reset_episode_cache()
