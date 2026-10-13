from evid_rl_env.environment.actions import ACTIONS, Actions

ROLES = ("support", "contradict")

# FINALIZE belongs to the environment, so a role never ends the debate itself.
SUPPORT_ROLE = [a for a in ACTIONS if a not in (Actions.CONTRADICT, Actions.FINALIZE)]
CONTRADICT_ROLE = [a for a in ACTIONS if a not in (Actions.SUPPORT, Actions.FINALIZE)]
_ROLE_ACTIONS = {"support": SUPPORT_ROLE, "contradict": CONTRADICT_ROLE}


class DebateEnv:
    """Alternate two roles over one ``ClaimEnv`` and finalize after a fixed number of turns.

    Both roles share the evidence pool, the selected set and the transcript.
    ``step`` takes the action of whichever role is up and returns that step's
    reward. If no confidence has been assigned by the last turn, that turn must
    assign one, so the final score does not fall back to the evidence-count
    guess. If ``max_tokens`` is set and the roles' payloads have used that many
    tokens, the debate ends early. It is still finalized, so the arbiter always
    scores the transcript and ``info["truncated"]`` marks the early end. After the last turn the environment runs FINALIZE itself, so the
    arbiter that backs the ``ClaimEnv`` scores the full transcript. That reward
    comes back in ``info["final_reward"]`` and belongs to both roles.
    """

    def __init__(self, env, turns_per_side: int = 5, max_tokens: int | None = None):
        if turns_per_side < 2:
            raise ValueError("turns_per_side must be at least 2")
        self.env = env
        self.turns_per_side = turns_per_side
        self.max_tokens = max_tokens
        self.tokens_used = 0
        self._turn = 0

    @property
    def total_turns(self) -> int:
        return 2 * self.turns_per_side

    @property
    def role(self) -> str:
        return self.role_at(self._turn)

    @staticmethod
    def role_at(turn: int) -> str:
        return ROLES[turn % 2]

    @staticmethod
    def allowed_actions(role: str) -> list:
        return _ROLE_ACTIONS[role]

    def current_allowed_actions(self) -> list:
        """Return the actions the role whose turn it is may take now."""
        state = self.env.state
        last_turn = self._turn == self.total_turns - 1
        # With nothing selected, ASSIGN_CONFIDENCE is not available to any policy,
        # so forcing it would leave the role with no action.
        if last_turn and state.confidence is None and state.selected_evidence:
            return [Actions.ASSIGN_CONFIDENCE]
        return self.allowed_actions(self.role)

    def reset(self):
        state = self.env.reset()
        # One step past the last turn, so the forced FINALIZE is not cut off by
        # the step limit, which would replace the arbiter's reward with -0.2.
        state.max_steps = self.total_turns + 1
        self._turn = 0
        self.tokens_used = 0
        return state

    def step(self, action, payload):
        if self._turn >= self.total_turns:
            raise RuntimeError("the debate is over, call reset()")
        if action not in self.current_allowed_actions():
            raise ValueError(f"{self.role} may not use {action}")

        state, reward, done, info = self.env.step(action, payload)
        self._turn += 1
        if isinstance(payload, dict):
            self.tokens_used += payload.get("tokens", 0)
        truncated = self.max_tokens is not None and self.tokens_used >= self.max_tokens
        if self._turn < self.total_turns and not truncated:
            return state, reward, done, info

        state, final_reward, done, final_info = self.env.step(Actions.FINALIZE, None)
        if not done:
            raise RuntimeError("FINALIZE did not end the debate")
        info = {**info, **final_info, "final_reward": final_reward, "truncated": truncated}
        return state, reward, done, info
