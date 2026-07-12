import numpy as np
import pytest

from evid_rl_env.environment.actions import Actions

GOOD_SCORES_JSON = (
    '{"LCS": 0.8, "ESS": 0.8, "GRS": 0.2, "COMP": 0.8, "BIAS": 0.2, "confidence": 0.9}'
)
# Distinct score set so tests can tell "judge was called again" apart from
# "cached score was reused" — the two must never look identical.
OTHER_SCORES_JSON = (
    '{"LCS": 0.1, "ESS": 0.1, "GRS": 0.1, "COMP": 0.1, "BIAS": 0.1, "confidence": 0.9}'
)


def test_reset_builds_evidence_pool_from_fetch(make_env, fake_evidence_docs):
    env = make_env()
    state = env.reset()
    assert len(state.evidence_pool) == len(fake_evidence_docs)
    assert [e.id for e in state.evidence_pool] == list(range(len(fake_evidence_docs)))


def test_select_new_evidence_gives_positive_reward(make_env):
    env = make_env()
    env.reset()
    _, reward, done, _ = env.step(Actions.SELECT, 0)
    assert not done
    assert reward > 0


def test_select_duplicate_evidence_is_penalized(make_env):
    env = make_env()
    env.reset()
    env.step(Actions.SELECT, 0)
    _, reward, _, _ = env.step(Actions.SELECT, 0)
    assert reward == pytest.approx(-0.1)


def test_remove_evidence_is_penalized(make_env):
    env = make_env()
    env.reset()
    env.step(Actions.SELECT, 0)
    _, reward, _, _ = env.step(Actions.REMOVE, 0)
    assert reward == pytest.approx(-0.05)


def test_query_within_budget_rewards_new_documents(make_env, fake_evidence_docs, fake_evidence_labeler):
    labels = {
        fake_evidence_docs[0]["content"]: "support",
        fake_evidence_docs[1]["content"]: "contradict",
        fake_evidence_docs[2]["content"]: "neutral",
    }
    env = make_env(evidence_labeler=fake_evidence_labeler(labels))
    env.reset()
    _, reward, _, _ = env.step(Actions.QUERY, "more evidence please")
    assert reward == pytest.approx(0.10)  # 2 useful (support+contradict) docs, below the 0.15 cap


def test_query_gives_no_reward_for_only_neutral_documents(make_env):
    """Root-cause regression test: QUERY reward used to be 0.05 * raw doc
    count regardless of label, so spamming queries for filler evidence paid
    off. It's now gated on support/contradict labels."""
    env = make_env()  # default labeler labels everything "neutral"
    env.reset()
    _, reward, _, _ = env.step(Actions.QUERY, "more evidence please")
    assert reward == 0.0


def test_query_over_budget_is_penalized(make_env):
    env = make_env()
    state = env.reset()
    assert state.max_queries == 2
    env.step(Actions.QUERY, "q1")
    env.step(Actions.QUERY, "q2")
    _, reward, _, _ = env.step(Actions.QUERY, "q3")
    assert reward == pytest.approx(-0.1)


def test_rerank_sorts_selected_evidence_by_claim_similarity(make_env, fake_evidence_docs):
    claim = "Test claim for the episode."
    # Deliberately NOT ranked by text length, so a passing test proves the sort
    # is driven by the (fake) embedder rather than the old length heuristic.
    vectors = {
        claim: np.array([1.0, 0.0]),
        fake_evidence_docs[0]["content"]: np.array([0.9, 0.1]),  # highest similarity
        fake_evidence_docs[1]["content"]: np.array([0.1, 0.9]),  # lowest similarity
        fake_evidence_docs[2]["content"]: np.array([0.5, 0.5]),  # middle similarity
    }
    fake_embedder = lambda text: vectors.get(text)  # noqa: E731

    env = make_env(dataset=[{"claim": claim, "search_query": claim}], embedder=fake_embedder)
    state = env.reset()
    for e in list(state.evidence_pool):
        env.step(Actions.SELECT, e.id)
    _, reward, _, _ = env.step(Actions.RERANK, None)

    assert [e.text for e in env.state.selected_evidence] == [
        fake_evidence_docs[0]["content"],
        fake_evidence_docs[2]["content"],
        fake_evidence_docs[1]["content"],
    ]
    assert reward == pytest.approx(0.02)


def test_rerank_falls_back_to_text_length_when_embedder_unavailable(make_env):
    env = make_env(embedder=lambda text: None)
    state = env.reset()
    for e in list(state.evidence_pool):
        env.step(Actions.SELECT, e.id)
    env.step(Actions.RERANK, None)
    lengths = [len(e.text) for e in env.state.selected_evidence]
    assert lengths == sorted(lengths, reverse=True)


def test_step_limit_terminates_episode(make_env):
    env = make_env()
    env.reset()
    done = False
    steps = 0
    while not done and steps < 20:
        _, reward, done, _ = env.step(Actions.RERANK, None)
        steps += 1
    assert done
    assert steps == 10  # State.max_steps default


# ---------------------------------------------------------------------------
# Judge-call cadence
# ---------------------------------------------------------------------------

def test_first_debate_action_at_step_one_reaches_the_judge(make_env):
    """Regression test: _last_judge_step used to start at 0, so steps_taken==1
    gave (1 - 0) < 2 and silently skipped the judge on the very first action."""
    env = make_env(mock_response=GOOD_SCORES_JSON)
    env.reset()
    _, _, _, info = env.step(Actions.SUPPORT, "opening argument")
    assert info["llm_reward"] == pytest.approx(0.518)
    assert env.state.last_llm_score == pytest.approx(0.518)


def test_judge_is_not_called_again_within_two_steps(make_env):
    env = make_env(responses=[GOOD_SCORES_JSON, OTHER_SCORES_JSON])
    env.reset()
    _, _, _, info1 = env.step(Actions.SUPPORT, "argument one")
    _, _, _, info2 = env.step(Actions.SUPPORT, "argument two")
    # If the gate incorrectly let a second call through, info2 would reflect
    # OTHER_SCORES_JSON instead of the reused first score.
    assert info1["llm_reward"] == pytest.approx(0.518)
    assert info2["llm_reward"] == pytest.approx(0.518)


def test_judge_is_called_again_after_two_steps(make_env):
    env = make_env(responses=[GOOD_SCORES_JSON, OTHER_SCORES_JSON])
    env.reset()
    env.step(Actions.SUPPORT, "argument one")       # step 1, calls judge
    env.step(Actions.RERANK, None)                   # step 2, no judge call
    _, _, _, info = env.step(Actions.SUPPORT, "argument two")  # step 3, gate reopens
    assert info["llm_reward"] == pytest.approx(0.0815)  # OTHER_SCORES_JSON's reward


# ---------------------------------------------------------------------------
# FINALIZE guards
# ---------------------------------------------------------------------------

def test_finalize_with_no_evidence_is_terminal_and_heavily_penalized(make_env):
    env = make_env()
    env.reset()
    _, reward, done, _ = env.step(Actions.FINALIZE, None)
    assert reward == pytest.approx(-1.0)
    assert done is True


def test_finalize_too_early_penalizes_but_does_not_end_episode(make_env):
    env = make_env()
    env.reset()
    env.step(Actions.SELECT, 0)   # step 1
    _, reward, done, _ = env.step(Actions.FINALIZE, None)  # step 2, steps_taken<=2
    assert reward == pytest.approx(-0.5)
    assert done is False
    # episode must still be usable afterwards
    _, _, done2, _ = env.step(Actions.SELECT, 1)
    assert done2 is False


def test_finalize_passes_llm_reward_into_base_reward(make_env, monkeypatch):
    """Regression test: base_reward used to be computed with llm_judge=None,
    silently redistributing the documented 0.10 llm-reward weight into F1/CA."""
    env = make_env(mock_response=GOOD_SCORES_JSON)
    env.reset()
    env.step(Actions.SELECT, 0)          # step 1
    env.step(Actions.SUPPORT, "argument")  # step 2

    captured = {}
    original_compute = env.reward_fn.compute

    def spy_compute(state, final_output, llm_reward=None):
        captured["llm_reward"] = llm_reward
        return original_compute(state, final_output, llm_reward=llm_reward)

    monkeypatch.setattr(env.reward_fn, "compute", spy_compute)

    _, _, done, info = env.step(Actions.FINALIZE, None)  # step 3

    assert done is True
    assert captured["llm_reward"] == pytest.approx(0.518)
    assert captured["llm_reward"] == info["llm_reward"]


def test_finalize_with_empty_debate_history_scores_worse_than_with_debate(make_env):
    """FINALIZE applies an explicit -0.3 penalty when debate_history is empty,
    compounded by the judge's own empty-reasoning short-circuit (reasoning is
    the joined debate_history, so the two are inherently coupled)."""
    no_debate_env = make_env()
    no_debate_env.reset()
    no_debate_env.step(Actions.SELECT, 0)
    no_debate_env.step(Actions.SELECT, 1)
    _, reward_no_debate, done, _ = no_debate_env.step(Actions.FINALIZE, None)
    assert done is True

    with_debate_env = make_env()
    with_debate_env.reset()
    with_debate_env.step(Actions.SELECT, 0)
    with_debate_env.step(Actions.SUPPORT, "an argument")
    _, reward_with_debate, done2, _ = with_debate_env.step(Actions.FINALIZE, None)
    assert done2 is True

    assert reward_with_debate > reward_no_debate


# ---------------------------------------------------------------------------
# Potential-based shaping: terminal Phi(s) = 0 convention
# ---------------------------------------------------------------------------

def test_shaping_uses_terminal_potential_zero_on_finalize(make_env):
    env = make_env(mock_response=GOOD_SCORES_JSON)
    env.reset()
    env.step(Actions.SELECT, 0)
    env.step(Actions.SUPPORT, "argument")
    assert env._prev_phi != 0.0  # judge fired, potential is nonzero mid-episode

    env.step(Actions.FINALIZE, None)
    assert env._prev_phi == 0.0  # terminal convention applied


def test_shaping_uses_terminal_potential_zero_on_step_limit(make_env):
    env = make_env(mock_response=GOOD_SCORES_JSON)
    env.reset()
    env.step(Actions.SUPPORT, "argument")  # step 1, judge fires
    assert env._prev_phi != 0.0

    done = False
    while not done:
        _, _, done, _ = env.step(Actions.RERANK, None)

    assert env._prev_phi == 0.0


# ---------------------------------------------------------------------------
# Evidence labeling
# ---------------------------------------------------------------------------

def test_reset_labels_evidence_via_injected_labeler(make_env, fake_evidence_docs, fake_evidence_labeler):
    labels = {
        fake_evidence_docs[0]["content"]: "support",
        fake_evidence_docs[1]["content"]: "contradict",
        fake_evidence_docs[2]["content"]: "adversarial",
    }
    env = make_env(evidence_labeler=fake_evidence_labeler(labels))
    state = env.reset()
    assert {e.text: e.label for e in state.evidence_pool} == labels


def test_query_labels_new_evidence_via_injected_labeler(make_env, fake_evidence_docs, fake_evidence_labeler):
    labels = {fake_evidence_docs[0]["content"]: "adversarial"}
    env = make_env(evidence_labeler=fake_evidence_labeler(labels, default="support"))
    env.reset()
    env.step(Actions.QUERY, "more evidence please")
    new_evidence = env.state.evidence_pool[len(fake_evidence_docs):]
    assert len(new_evidence) == len(fake_evidence_docs)
    assert {e.text: e.label for e in new_evidence} == {
        doc["content"]: labels.get(doc["content"], "support") for doc in fake_evidence_docs
    }


def test_finalize_reward_reflects_injected_evidence_labels(make_env, fake_evidence_docs, fake_evidence_labeler):
    """Root-cause regression test: before evidence labeling existed, every
    document defaulted to "neutral" forever, which made base_reward's F1
    always 0, contradiction-acknowledgment always the vacuous 1.0, and
    adversarial-contamination always 0 — three of five terms were frozen
    constants regardless of what the agent actually selected. With real
    per-evidence labels, the same selection now produces a different reward."""
    labels = {
        fake_evidence_docs[0]["content"]: "support",
        fake_evidence_docs[1]["content"]: "contradict",
        fake_evidence_docs[2]["content"]: "adversarial",
    }

    def _finalize_reward(labeler):
        env = make_env(evidence_labeler=labeler)
        state = env.reset()
        for e in list(state.evidence_pool):
            env.step(Actions.SELECT, e.id)
        env.step(Actions.SUPPORT, "argument")
        _, reward, _, _ = env.step(Actions.FINALIZE, None)
        return reward

    reward_all_neutral = _finalize_reward(fake_evidence_labeler())
    reward_with_labels = _finalize_reward(fake_evidence_labeler(labels))

    assert reward_with_labels != pytest.approx(reward_all_neutral)


# ---------------------------------------------------------------------------
# New actions: ASSIGN_CONFIDENCE, CHALLENGE_EVIDENCE, REQUEST_CLARIFICATION, HEDGE
# ---------------------------------------------------------------------------

def test_assign_confidence_sets_state_and_is_rewarded(make_env):
    env = make_env()
    env.reset()
    env.step(Actions.SELECT, 0)
    _, reward, _, _ = env.step(Actions.ASSIGN_CONFIDENCE, 0.9)
    assert reward == pytest.approx(0.05)
    assert env.state.confidence == pytest.approx(0.9)


def test_assign_confidence_clamps_out_of_range_values(make_env):
    env = make_env()
    env.reset()
    env.step(Actions.ASSIGN_CONFIDENCE, 5.0)
    assert env.state.confidence == pytest.approx(1.0)


def test_assign_confidence_with_invalid_payload_is_noop(make_env):
    env = make_env()
    env.reset()
    _, reward, _, _ = env.step(Actions.ASSIGN_CONFIDENCE, "not-a-number")
    assert reward == 0.0
    assert env.state.confidence is None


def test_assign_confidence_overrides_heuristic_at_finalize(make_env, monkeypatch):
    """Regression guard: FINALIZE must use the agent's own ASSIGN_CONFIDENCE
    value (when set) instead of the min(1, n_selected/3) heuristic."""
    env = make_env()
    env.reset()
    env.step(Actions.SELECT, 0)          # step 1
    env.step(Actions.ASSIGN_CONFIDENCE, 0.83)  # step 2

    captured = {}
    original_compute = env.reward_fn.compute

    def spy_compute(state, final_output, llm_reward=None):
        captured["confidence"] = final_output["confidence"]
        return original_compute(state, final_output, llm_reward=llm_reward)

    monkeypatch.setattr(env.reward_fn, "compute", spy_compute)

    env.step(Actions.SUPPORT, "argument")  # step 3
    env.step(Actions.FINALIZE, None)       # step 4

    assert captured["confidence"] == pytest.approx(0.83)


def test_challenge_evidence_bonus_for_targeting_adversarial_evidence(
    make_env, fake_evidence_docs, fake_evidence_labeler
):
    labels = {fake_evidence_docs[0]["content"]: "adversarial"}
    env = make_env(mock_response=GOOD_SCORES_JSON,
                    evidence_labeler=fake_evidence_labeler(labels, default="neutral"))
    env.reset()
    _, reward, _, info = env.step(
        Actions.CHALLENGE_EVIDENCE, {"evidence_id": 0, "argument": "unreliable source"}
    )
    assert info["llm_reward"] == pytest.approx(0.518)
    # base debate reward + adversarial-target bonus + potential-based shaping
    # (phi_prev=0 on the first step, phi_next=llm_reward since the episode
    # isn't done yet: shaping = 0.1 * (0.99 * 0.518 - 0))
    assert reward == pytest.approx(0.05 + 0.15 * 0.518 + 0.1 + 0.1 * (0.99 * 0.518))
    assert 0 in env.state.challenged_evidence_ids
    assert "CHALLENGE(0): unreliable source" in env.state.debate_history


def test_challenge_evidence_penalty_for_targeting_supporting_evidence(
    make_env, fake_evidence_docs, fake_evidence_labeler
):
    labels = {fake_evidence_docs[0]["content"]: "support"}
    env = make_env(mock_response=GOOD_SCORES_JSON,
                    evidence_labeler=fake_evidence_labeler(labels, default="neutral"))
    env.reset()
    _, reward, _, _ = env.step(
        Actions.CHALLENGE_EVIDENCE, {"evidence_id": 0, "argument": "unreliable source"}
    )
    assert reward == pytest.approx(0.05 + 0.15 * 0.518 - 0.05 + 0.1 * (0.99 * 0.518))


def test_challenge_evidence_with_unknown_target_is_noop(make_env):
    env = make_env()
    env.reset()
    _, reward, _, _ = env.step(
        Actions.CHALLENGE_EVIDENCE, {"evidence_id": 999, "argument": "x"}
    )
    assert reward == 0.0
    assert env.state.debate_history == []


def test_request_clarification_within_budget_fetches_new_evidence(make_env, fake_evidence_docs, fake_evidence_labeler):
    labels = {
        fake_evidence_docs[0]["content"]: "support",
        fake_evidence_docs[1]["content"]: "contradict",
        fake_evidence_docs[2]["content"]: "neutral",
    }
    env = make_env(evidence_labeler=fake_evidence_labeler(labels))
    env.reset()
    _, reward, _, _ = env.step(Actions.REQUEST_CLARIFICATION, "narrower query")
    assert reward == pytest.approx(0.10)  # 2 useful (support+contradict) docs, below the 0.15 cap
    assert env.state.query_count == 1
    assert "CLARIFY: narrower query" in env.state.debate_history


def test_request_clarification_shares_budget_with_query(make_env):
    env = make_env()
    state = env.reset()
    assert state.max_queries == 2
    env.step(Actions.QUERY, "q1")
    env.step(Actions.REQUEST_CLARIFICATION, "q2")
    _, reward, _, _ = env.step(Actions.REQUEST_CLARIFICATION, "q3")
    assert reward == pytest.approx(-0.1)


def test_hedge_triggers_judge_and_reward_like_concede(make_env):
    env = make_env(mock_response=GOOD_SCORES_JSON)
    env.reset()
    _, reward, _, info = env.step(Actions.HEDGE, {"argument": "partially true"})
    assert info["llm_reward"] == pytest.approx(0.518)
    assert reward == pytest.approx(0.05 + 0.1 * 0.518 + 0.1 * (0.99 * 0.518))
    assert "HEDGE: partially true" in env.state.debate_history


# ---------------------------------------------------------------------------
# task_success info signal (curriculum performance signal)
# ---------------------------------------------------------------------------

def test_finalize_includes_bounded_task_success_in_info(make_env):
    env = make_env()
    env.reset()
    env.step(Actions.SELECT, 0)
    env.step(Actions.SUPPORT, "argument")
    _, _, done, info = env.step(Actions.FINALIZE, None)
    assert done is True
    assert info["task_success"] is not None
    assert 0.0 <= info["task_success"] <= 1.0


def test_non_finalize_steps_have_no_task_success(make_env):
    env = make_env()
    env.reset()
    _, _, _, info = env.step(Actions.SELECT, 0)
    assert info["task_success"] is None


# ---------------------------------------------------------------------------
# judge_ensemble_models dispatch (Phase 2: multi-judge ensemble)
# ---------------------------------------------------------------------------

def test_judge_ensemble_models_dispatches_to_ensemble_judge(monkeypatch):
    """ClaimEnv(judge_ensemble_models=[...]) must use _get_ensemble_judge,
    not the single-judge _get_llm_judge path — monkeypatched so this never
    loads a real model."""
    from evid_rl_env.environment import environment as env_module

    sentinel = object()
    captured = {}

    def _fake_get_ensemble_judge(model_names, seed):
        captured["model_names"] = model_names
        captured["seed"] = seed
        return sentinel

    def _fail_get_llm_judge(model_name, seed):
        raise AssertionError("single-judge path should not be used when judge_ensemble_models is set")

    monkeypatch.setattr(env_module, "_get_ensemble_judge", _fake_get_ensemble_judge)
    monkeypatch.setattr(env_module, "_get_llm_judge", _fail_get_llm_judge)

    env = env_module.ClaimEnv(
        [{"id": "c1", "claim": "test claim", "label": 1.0}],
        seed=7,
        judge_ensemble_models=["model-a", "model-b"],
        evidence_labeler=object(),  # avoid _get_evidence_labeler's model load too
    )

    assert env.llm_judge is sentinel
    assert captured["model_names"] == ("model-a", "model-b")
    assert captured["seed"] == 7


def test_judge_escalation_dispatches_to_escalating_judge(monkeypatch):
    """ClaimEnv(judge_ensemble_models=[...], judge_escalation=True) must use
    _get_escalating_judge, not the always-on _get_ensemble_judge path —
    monkeypatched so this never loads a real model."""
    from evid_rl_env.environment import environment as env_module

    sentinel = object()
    captured = {}

    def _fake_get_escalating_judge(judge_model_name, ensemble_models, seed, escalation_target="ensemble"):
        captured["judge_model_name"] = judge_model_name
        captured["ensemble_models"] = ensemble_models
        captured["seed"] = seed
        captured["escalation_target"] = escalation_target
        return sentinel

    def _fail_get_ensemble_judge(model_names, seed):
        raise AssertionError("always-on ensemble path should not be used when judge_escalation is True")

    monkeypatch.setattr(env_module, "_get_escalating_judge", _fake_get_escalating_judge)
    monkeypatch.setattr(env_module, "_get_ensemble_judge", _fail_get_ensemble_judge)

    env = env_module.ClaimEnv(
        [{"id": "c1", "claim": "test claim", "label": 1.0}],
        seed=7,
        judge_model="model-cheap",
        judge_ensemble_models=["model-a", "model-b"],
        judge_escalation=True,
        evidence_labeler=object(),
    )

    assert env.llm_judge is sentinel
    assert captured["judge_model_name"] == "model-cheap"
    assert captured["ensemble_models"] == ("model-a", "model-b")
    assert captured["seed"] == 7


def test_judge_escalation_false_still_uses_always_on_ensemble(monkeypatch):
    """Regression check: judge_ensemble_models alone (judge_escalation
    unset/False) must keep dispatching to _get_ensemble_judge exactly as it
    did before this phase."""
    from evid_rl_env.environment import environment as env_module

    sentinel = object()

    def _fake_get_ensemble_judge(model_names, seed):
        return sentinel

    def _fail_get_escalating_judge(*args, **kwargs):
        raise AssertionError("escalation path should not be used when judge_escalation is False")

    monkeypatch.setattr(env_module, "_get_ensemble_judge", _fake_get_ensemble_judge)
    monkeypatch.setattr(env_module, "_get_escalating_judge", _fail_get_escalating_judge)

    env = env_module.ClaimEnv(
        [{"id": "c1", "claim": "test claim", "label": 1.0}],
        judge_ensemble_models=["model-a", "model-b"],
        evidence_labeler=object(),
    )

    assert env.llm_judge is sentinel


def test_get_escalating_judge_debate_target_uses_debate_judge(monkeypatch):
    """_get_escalating_judge(..., escalation_target="debate") must call
    _get_debate_judge, not _get_ensemble_judge — monkeypatched so this
    never loads a real model."""
    from evid_rl_env.environment import environment as env_module

    cheap_sentinel = object()
    debate_sentinel = object()

    monkeypatch.setattr(env_module, "_get_llm_judge", lambda name, seed: cheap_sentinel)
    monkeypatch.setattr(env_module, "_get_debate_judge", lambda seed: debate_sentinel)

    def _fail_get_ensemble_judge(model_names, seed):
        raise AssertionError("ensemble path should not be used for escalation_target='debate'")

    monkeypatch.setattr(env_module, "_get_ensemble_judge", _fail_get_ensemble_judge)

    result = env_module._get_escalating_judge(
        "model-cheap", ("model-a", "model-b"), seed=7, escalation_target="debate",
    )

    assert result.cheap_judge is cheap_sentinel
    assert result.escalated_judge is debate_sentinel


def test_get_escalating_judge_ensemble_target_still_uses_ensemble_judge(monkeypatch):
    """Regression check: the default escalation_target='ensemble' must keep
    dispatching to _get_ensemble_judge exactly as Phase 3 left it."""
    from evid_rl_env.environment import environment as env_module

    cheap_sentinel = object()
    ensemble_sentinel = object()

    monkeypatch.setattr(env_module, "_get_llm_judge", lambda name, seed: cheap_sentinel)
    monkeypatch.setattr(env_module, "_get_ensemble_judge", lambda model_names, seed: ensemble_sentinel)

    def _fail_get_debate_judge(seed):
        raise AssertionError("debate path should not be used for escalation_target='ensemble'")

    monkeypatch.setattr(env_module, "_get_debate_judge", _fail_get_debate_judge)

    result = env_module._get_escalating_judge(
        "model-cheap", ("model-a", "model-b"), seed=7,
    )

    assert result.cheap_judge is cheap_sentinel
    assert result.escalated_judge is ensemble_sentinel


def test_claim_env_forwards_judge_escalation_target(monkeypatch):
    from evid_rl_env.environment import environment as env_module

    sentinel = object()
    captured = {}

    def _fake_get_escalating_judge(judge_model_name, ensemble_models, seed, escalation_target="ensemble"):
        captured["escalation_target"] = escalation_target
        return sentinel

    monkeypatch.setattr(env_module, "_get_escalating_judge", _fake_get_escalating_judge)

    env = env_module.ClaimEnv(
        [{"id": "c1", "claim": "test claim", "label": 1.0}],
        judge_ensemble_models=["model-a", "model-b"],
        judge_escalation=True,
        judge_escalation_target="debate",
        evidence_labeler=object(),
    )

    assert env.llm_judge is sentinel
    assert captured["escalation_target"] == "debate"


def test_llm_judge_override_takes_priority_over_ensemble(monkeypatch):
    """The existing test-injection seam (llm_judge=) must still win even
    when judge_ensemble_models is also set."""
    from evid_rl_env.environment import environment as env_module

    def _fail(*args, **kwargs):
        raise AssertionError("should not be called when llm_judge= is provided")

    monkeypatch.setattr(env_module, "_get_ensemble_judge", _fail)
    monkeypatch.setattr(env_module, "_get_llm_judge", _fail)

    injected = object()
    env = env_module.ClaimEnv(
        [{"id": "c1", "claim": "test claim", "label": 1.0}],
        llm_judge=injected,
        judge_ensemble_models=["model-a", "model-b"],
        evidence_labeler=object(),
    )

    assert env.llm_judge is injected
