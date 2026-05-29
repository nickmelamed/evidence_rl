import numpy as np
from evid_rl_env.environment.actions import Actions, ACTIONS
from evid_rl_env.agent.llm_client import LLMClient
from evid_rl_env.agent.state_encoder import encode_state_semantic, get_state_dim, SEMANTIC_AVAILABLE

STATE_DIM = get_state_dim()

def encode_state(state):
    return encode_state_semantic(state)

class ActorCriticPolicy:
    def __init__(self, n_actions, state_dim=None):
        if state_dim is None: state_dim = get_state_dim()
        self.actions = ACTIONS
        self.n_actions = n_actions
        self.state_dim = state_dim

        # Actor
        self.actor_params = np.random.randn(state_dim, n_actions) * 0.01

        # Critic (value function)
        self.value_params = np.zeros(state_dim)

        self.llm = LLMClient()

    def generate_arguments_batch(self, prompts):
        """Generate multiple arguments in a single batched pipeline call."""
        if not prompts:
            return []
        outputs = self.llm.pipe(
            prompts,
            batch_size=min(len(prompts), 4),
            max_new_tokens=128,
            do_sample=True,
            temperature=self.llm.temperature,
            return_full_text=False
        )
        return [(o[0]["generated_text"], len(o[0]["generated_text"].split())) for o in outputs]

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

    def act(self, state):
        if not hasattr(self, "_arg_cache"):
            self._arg_cache = {}

        probs = self.get_probs(state)

        entropy = -np.sum(probs * np.log(probs + 1e-8))
        self.last_entropy = entropy

        # mask QUERY when budget exhausted
        if state.query_count >= state.max_queries:
            probs = probs.copy()
            query_idx = self.actions.index(Actions.QUERY)
            probs[query_idx] = 0
            probs = probs / np.sum(probs)

        self.last_probs = probs.copy()

        idx = np.random.choice(self.n_actions, p=probs)
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

    def grad_log_prob(self, state, action_idx):
        probs = self.get_probs(state)
        grad = -probs
        grad[action_idx] += 1

        features = encode_state(state)
        return np.outer(features, grad)

class SoftmaxPolicy:
    def __init__(self, n_actions, state_dim=None):
        if state_dim is None: state_dim = get_state_dim()
        self.actions = ACTIONS
        self.n_actions = n_actions
        self.params = np.random.randn(state_dim, n_actions)

        finalize_idx = self.actions.index(Actions.FINALIZE)
        self.params[finalize_idx] = -2.0

        self.llm = LLMClient()

    def get_logits(self, state):
        features = encode_state(state)
        return features @ self.params

    def get_probs(self, state):
        logits = self.get_logits(state)
        logits = logits - np.max(logits) # numeric stability
        exp = np.exp(logits)
        return exp / np.sum(exp)


    def act(self, state):
        probs = self.get_probs(state).copy() # make it writable

        entropy = -np.sum(probs * np.log(probs + 1e-8))
        self.last_entropy = entropy

        # mask FINALIZE early
        if state.steps_taken < 2:
            finalize_idx = self.actions.index(Actions.FINALIZE)
            probs[finalize_idx] = 0
            probs = probs / np.sum(probs)

        self.last_probs = probs.copy()

        # epsilon-greedy exploration
        if np.random.rand() < 0.3:
            idx = np.random.choice(self.n_actions)
        else:
            idx = np.random.choice(self.n_actions, p=probs)

        action = self.actions[idx]

        # payload logic
        if action == Actions.SELECT:
            doc = np.random.choice(state.evidence_pool)
            return action, doc.id

        elif action in [Actions.SUPPORT, Actions.CONTRADICT]:

            selected_texts = [
                e.text for e in state.selected_evidence
            ]

            argument = self.llm.generate(
                f"""
            Claim: {state.claim}

            Evidence:
            {selected_texts}

            Write a concise {action.lower()} argument.
            """
            )

            return action, {
                "argument": argument,
                "evidence_ids": [e.id for e in state.selected_evidence]
            }

        elif action == Actions.QUERY:
            query_prompt = f"Given claim: {state.claim}\nWrite a short search query to find more relevant evidence."
            query_text, tokens = self.llm.generate(query_prompt)
            return action, query_text.strip()

        elif action == Actions.RERANK:
            return action, None

        elif action == Actions.SUMMARIZE:
            if state.selected_evidence:
                texts = [e.text for e in state.selected_evidence]
                summary_prompt = f"Claim: {state.claim}\nEvidence:\n{texts}\nWrite a one-sentence summary of how this evidence relates to the claim."
                summary_text, tokens = self.llm.generate(summary_prompt)
                return action, {"summary": summary_text.strip(), "tokens": tokens}
            return action, {"summary": "", "tokens": 0}

        elif action == Actions.CONCEDE:
            texts = [e.text for e in state.selected_evidence]
            concede_prompt = f"Claim: {state.claim}\nEvidence: {texts}\nWrite a concise acknowledgement of the strongest counterpoint to the claim."
            argument, tokens = self.llm.generate(concede_prompt)
            return action, {"argument": argument, "evidence_ids": [e.id for e in state.selected_evidence], "tokens": tokens}

        return action, None

    def grad_log_prob(self, state, action_idx):
        probs = self.get_probs(state)
        grad = -probs
        grad[action_idx] += 1
        return grad