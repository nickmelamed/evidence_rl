import csv
import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "build_results_tables", Path(__file__).parent.parent / "scripts" / "build_results_tables.py"
)
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)


def _make_run(root, name, cfg, rewards, gold=None, evals=None):
    run = root / name
    run.mkdir()
    (run / "config.json").write_text(json.dumps(cfg))
    with (run / "metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward"])
        writer.writerows(enumerate(rewards))
    if gold:
        (run / "gold_eval.jsonl").write_text("\n".join(json.dumps(g) for g in gold))
    if evals:
        (run / "eval_results.json").write_text(json.dumps(evals))
    return run


def test_judge_architecture_names():
    assert build.judge_architecture({}) == "single"
    assert build.judge_architecture({"judge_ensemble_models": ["a", "b"]}) == "ensemble"
    assert build.judge_architecture(
        {"judge_ensemble_models": ["a", "b"], "judge_escalation": True,
         "judge_escalation_target": "debate"}
    ) == "escalating_debate"


def test_training_summary_uses_last_fifty_rewards(tmp_path):
    rewards = [0.0] * 50 + [1.0] * 50
    run = _make_run(tmp_path, "r1", {"algo": "ppo", "seed": 1}, rewards)
    row = build.training_rows([run])[0]
    assert row["mean_reward_all"] == 0.5
    assert row["mean_reward_last_50"] == 1.0
    assert row["run"] == "r1"


def test_gold_rows_use_last_round_and_keep_missing_values_empty(tmp_path):
    gold = [
        {"episode": 10, "n_episodes": 5, "n_scored": 0, "proxy_gold_correlation": None},
        {"episode": 20, "n_episodes": 5, "n_scored": 3, "proxy_gold_correlation": 0.123456},
    ]
    run = _make_run(tmp_path, "r2", {"algo": "ppo"}, [0.0], gold=gold)
    row = build.gold_rows([run])[0]
    assert row["rounds"] == 2
    assert row["n_scored"] == 3
    assert row["proxy_gold_correlation"] == 0.1235
    assert row["gold_reward_mean"] == ""


def test_runs_without_a_file_are_skipped_not_filled(tmp_path):
    run = _make_run(tmp_path, "r3", {}, [0.5])
    assert build.gold_rows([run]) == []
    assert build.eval_rows([run]) == []


def test_eval_rows_list_every_policy(tmp_path):
    evals = {"n_episodes": 50, "rl_win_rate_vs_greedy": 1.0,
             "baselines": {"rl": {"mean": 0.5, "std": 0.1}, "random": {"mean": 0.0, "std": 0.2}}}
    run = _make_run(tmp_path, "r4", {}, [0.5], evals=evals)
    rows = build.eval_rows([run])
    assert [r["policy"] for r in rows] == ["rl", "random"]
    assert rows[0]["reward_mean"] == 0.5


def test_summary_groups_seeds_and_reports_spread(tmp_path):
    runs = []
    for seed, last in ((1, 0.2), (2, 0.4), (3, 0.6)):
        cfg = {"algo": "ppo", "seed": seed}
        runs.append(_make_run(
            tmp_path, f"ppo_s{seed}", cfg, [last] * 50,
            gold=[{"episode": 10, "n_episodes": 5, "n_scored": 2,
                   "proxy_gold_correlation": 0.5 if seed != 3 else None}],
            evals={"n_episodes": 5, "baselines": {"rl": {"mean": last, "std": 0.1},
                                                   "greedy_llm": {"mean": 0.5, "std": 0.1}}},
        ))
    rows = build.summary_rows(build.training_rows(runs), build.eval_rows(runs), build.gold_rows(runs))
    assert len(rows) == 1
    row = rows[0]
    assert row["n_runs"] == 3
    assert row["train_reward_last_50_mean"] == 0.4
    assert row["train_reward_last_50_std"] == 0.2
    assert row["eval_rl_reward_mean"] == 0.4
    assert row["eval_greedy_llm_reward_mean"] == 0.5
    assert row["gold_runs_with_correlation"] == 2
    assert row["gold_n_scored_total"] == 6


def test_summary_spread_is_empty_for_a_single_run(tmp_path):
    run = _make_run(tmp_path, "solo", {"algo": "ppo", "seed": 1}, [0.3] * 5)
    rows = build.summary_rows(build.training_rows([run]), [], [])
    assert rows[0]["train_reward_last_50_std"] == ""
