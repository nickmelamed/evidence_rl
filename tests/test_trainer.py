from evid_rl_env.config import build_trainer


def test_trainer_runs():
    trainer = build_trainer({"episodes": 2})

    results = trainer.train()

    assert results is not None