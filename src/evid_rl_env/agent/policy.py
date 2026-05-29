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
        probs = self.get_probs(state)

        entropy = -np.sum(probs * np.log(probs + 1e-8))
        self.last_entropy = entropy
        self.last_probs = probs.copy()

        idx = np.random.choice(self.n_actions, p=probs)
        action = self.actions[idx]

        if action == Actions.SELECT:
            doc = np.random.choice(state.evidence_pool)
            return action, doc.id, idx

        elif action in [Actions.SUPPORT, Actions.CONTRADICT]:
            texts = [e.text for e in state.selected_evidence]

            argument, tokens = self.llm.generate(
                f"""
Claim: {state.claim}
Evidence: {texts}
Write a concise {action.lower()} argument.
"""
            )

            return action, {
                "argument": argument,
                "evidence_ids": [e.id for e in state.selected_evidence],
                "tokens": tokens
            }, idx

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

        return action, None

    def grad_log_prob(self, state, action_idx):
        probs = self.get_probs(state)
        grad = -probs
        grad[action_idx] += 1
        return grad