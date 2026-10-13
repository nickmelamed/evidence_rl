import numpy as np
import pytest

from evid_rl_env.agent.state_encoder import encode_state_semantic
from evid_rl_env.environment.actions import ACTIONS, Actions
from evid_rl_env.environment.debate_env import CONTRADICT_ROLE, SUPPORT_ROLE, DebateEnv
from evid_rl_env.environment.environment import ClaimEnv

DATASET = [{
    "claim": "Debate test claim.",
    "search_query": "debate test claim",
    "label": 1,
    "evidence": [
        {"text": "A randomized trial found a benefit.", "label": "support"},
        {"text": "A cohort study found no effect.", "label": "contradict"},
        {"text": "An unrelated background article.", "label": "neutral"},
    ],
}]
ARGUMENT = {"argument": "an argument", "evidence_ids": [0], "tokens": 3}


def _script(debate):
    """Select evidence, set a confidence, then argue until the debate ends."""
    steps = [(Actions.SELECT, 0), (Actions.SELECT, 1), (Actions.ASSIGN_CONFIDENCE, 0.8)]
    out = []
    for i in range(debate.turns_per_side * 2):
        if i < len(steps):
            out.append(steps[i])
        else:
            action = Actions.SUPPORT if debate.role_at(i) == "support" else Actions.CONTRADICT
            out.append((action, ARGUMENT))
    return out


def _play(debate):
    debate.reset()
    log = []
    done = False
    for action, payload in _script(debate):
        role = debate.role
        _, reward, done, info = debate.step(action, payload)
        log.append((role, action, reward, done))
    return log, info, debate.env.state


@pytest.fixture
def debate(make_env):
    return DebateEnv(make_env(dataset=DATASET), turns_per_side=5)


def test_roles_alternate_starting_with_support(debate):
    debate.reset()
    roles = []
    for action, payload in _script(debate)[:-1]:
        roles.append(debate.role)
        debate.step(action, payload)
    assert roles == ["support", "contradict"] * 4 + ["support"]


def test_the_episode_ends_after_the_fixed_turns_with_a_forced_finalize(debate):
    log, info, state = _play(debate)
    assert [done for *_, done in log] == [False] * 9 + [True]
    assert len(log) == 10
    assert state.steps_taken == 11
    assert "final_reward" in info
    assert info["task_success"] is not None


def test_a_role_cannot_finalize_or_use_the_other_sides_argument(debate):
    assert Actions.FINALIZE not in SUPPORT_ROLE
    assert Actions.FINALIZE not in CONTRADICT_ROLE
    assert Actions.CONTRADICT not in SUPPORT_ROLE
    assert Actions.SUPPORT not in CONTRADICT_ROLE
    assert set(SUPPORT_ROLE) | set(CONTRADICT_ROLE) == set(ACTIONS) - {Actions.FINALIZE}
    debate.reset()
    assert debate.allowed_actions("support") == SUPPORT_ROLE
    assert debate.allowed_actions("contradict") == CONTRADICT_ROLE


def test_stepping_with_a_blocked_action_raises(debate):
    debate.reset()
    with pytest.raises(ValueError):
        debate.step(Actions.CONTRADICT, ARGUMENT)


def test_the_arbiter_scores_the_two_sided_transcript(make_llm_judge, fake_evidence_labeler):
    judge = make_llm_judge()
    prompts = []
    original = judge.llm.generate_structured
    judge.llm.generate_structured = lambda prompt: (prompts.append(prompt), original(prompt))[1]
    env = ClaimEnv(DATASET, llm_judge=judge, seed=0, evidence_labeler=fake_evidence_labeler())
    debate = DebateEnv(env, turns_per_side=5)
    _play(debate)
    assert "SUPPORT: an argument" in prompts[-1]
    assert "CONTRADICT: an argument" in prompts[-1]


def test_same_seed_gives_the_same_transcript_and_rewards(make_env):
    first = _play(DebateEnv(make_env(dataset=DATASET, seed=3), turns_per_side=5))
    second = _play(DebateEnv(make_env(dataset=DATASET, seed=3), turns_per_side=5))
    assert first[0] == second[0]
    assert first[1]["final_reward"] == second[1]["final_reward"]
    assert first[2].debate_history == second[2].debate_history


def test_the_observation_does_not_depend_on_evidence_labels(debate):
    state = debate.reset()
    before = encode_state_semantic(state)
    for e in state.evidence_pool:
        e.label = "support"
    after = encode_state_semantic(state)
    np.testing.assert_array_equal(before, after)


def test_too_few_turns_is_rejected(make_env):
    with pytest.raises(ValueError):
        DebateEnv(make_env(dataset=DATASET), turns_per_side=1)


def _run_turns(debate, n, action=Actions.SELECT):
    """Play ``n`` turns of selecting evidence 0, 1, 2 in order."""
    for i in range(n):
        debate.step(action, i % 3)


def test_the_last_turn_must_assign_a_confidence_when_none_was_set(debate):
    debate.reset()
    _run_turns(debate, 9)
    assert debate.role == "contradict"
    assert debate.current_allowed_actions() == [Actions.ASSIGN_CONFIDENCE]
    with pytest.raises(ValueError):
        debate.step(Actions.CONTRADICT, ARGUMENT)
    _, _, done, info = debate.step(Actions.ASSIGN_CONFIDENCE, 0.7)
    assert done
    assert debate.env.state.confidence == 0.7


def test_no_confidence_is_forced_once_one_was_assigned(debate):
    debate.reset()
    debate.step(Actions.SELECT, 0)
    debate.step(Actions.ASSIGN_CONFIDENCE, 0.4)
    _run_turns(debate, 7)
    assert debate.current_allowed_actions() == CONTRADICT_ROLE


def test_no_confidence_is_forced_when_no_evidence_was_selected(debate):
    debate.reset()
    for _ in range(9):
        debate.step(Actions.RERANK, None)
    assert debate.current_allowed_actions() == CONTRADICT_ROLE


def test_a_spent_token_budget_ends_the_debate_with_a_scored_finalize(make_env):
    debate = DebateEnv(make_env(dataset=DATASET), turns_per_side=5, max_tokens=10)
    debate.reset()
    results = []
    for action, payload in _script(debate):
        results.append(debate.step(action, payload))
        if results[-1][2]:
            break

    # Three setup turns carry no tokens, then each argument adds 3, so the fourth
    # argument (turn 7) reaches 12 and ends the debate.
    assert len(results) == 7
    _, _, done, info = results[-1]
    assert done
    assert info["truncated"] is True
    assert "final_reward" in info
    assert debate.env.state.steps_taken == 8


def test_a_debate_inside_its_budget_is_not_truncated(make_env):
    debate = DebateEnv(make_env(dataset=DATASET), turns_per_side=5, max_tokens=1000)
    _, info, _ = _play(debate)
    assert info["truncated"] is False


def test_no_budget_means_no_truncation(debate):
    _, info, _ = _play(debate)
    assert info["truncated"] is False


def test_the_budget_resets_between_episodes(make_env):
    debate = DebateEnv(make_env(dataset=DATASET), turns_per_side=5, max_tokens=1000)
    _play(debate)
    debate.reset()
    assert debate.tokens_used == 0
