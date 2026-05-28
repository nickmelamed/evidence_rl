from evid_rl_env.environment.environment import Environment


def test_environment_step():
    env = Environment(max_steps=5)

    state = env.reset()
    assert state is not None

    next_state, reward, done, info = env.step(action=None)

    assert isinstance(done, bool)