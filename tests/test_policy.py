from evid_rl_env.agent.policy import Policy


def test_policy_action():
    policy = Policy()

    action = policy.act(state={"dummy": True})

    assert action is not None