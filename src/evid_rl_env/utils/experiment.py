import os
import json
import csv
from datetime import datetime


FIXED_FIELDS = [
    "episode",
    "reward",
    "num_steps",
    "final_decision",
    "correct",
    "num_selected",
    "num_removed",
    "num_support_actions",
    "num_contradict_actions",
    "entropy",
    "tokens",
    "policy_type",
    "algo"
]


class ExperimentTracker:
    def __init__(self, exp_name="default"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_name = f"{exp_name}_{timestamp}"

        self.base_dir = os.path.join("artifacts", "experiments", self.exp_name)
        self.traj_dir = os.path.join(self.base_dir, "trajectories")

        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.traj_dir, exist_ok=True)

        self.csv_path = os.path.join(self.base_dir, "metrics.csv")

        self._init_csv()

    def _init_csv(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIXED_FIELDS)
                writer.writeheader()

    def save_config(self, config: dict):
        with open(os.path.join(self.base_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def log_episode(self, metrics: dict):
        # enforce schema
        row = {k: metrics.get(k, None) for k in FIXED_FIELDS}

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIXED_FIELDS)
            writer.writerow(row)

    def save_trajectory(self, episode: int, trajectory: list):
        path = os.path.join(self.traj_dir, f"episode_{episode}.json")
        with open(path, "w") as f:
            json.dump(trajectory, f, indent=2)

    def save_summary(self):
        import pandas as pd

        df = pd.read_csv(self.csv_path)

        summary = {
            "total_episodes": len(df),
            "avg_reward": df["reward"].mean(),
            "max_reward": df["reward"].max(),
        }

        with open(os.path.join(self.base_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    def get_dir(self):
        return self.base_dir