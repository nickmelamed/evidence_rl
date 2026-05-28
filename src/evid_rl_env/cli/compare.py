import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse


def compare_experiments(exp_paths):
    plt.figure()

    for path in exp_paths:
        csv_path = os.path.join(path, "metrics.csv")

        if not os.path.exists(csv_path):
            print(f"Skipping {path} (no metrics.csv)")
            continue

        df = pd.read_csv(csv_path)
        label = os.path.basename(path)

        df["reward_smooth"] = df["reward"].rolling(window=5, min_periods=1).mean()

        plt.plot(df["episode"], df["reward_smooth"], label=label)

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Experiment Comparison")
    plt.legend()

    out_dir = "artifacts/experiments"
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "comparison.png")
    plt.savefig(out_path)

    print(f"Saved comparison plot to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", nargs="+", required=True)
    args = parser.parse_args()

    compare_experiments(args.paths)


if __name__ == "__main__":
    main()