import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse


def plot_experiment(exp_path):
    csv_path = os.path.join(exp_path, "metrics.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No metrics.csv in {exp_path}")

    df = pd.read_csv(csv_path)

    # Smooth reward (rolling avg)
    df["reward_smooth"] = df["reward"].rolling(window=5, min_periods=1).mean()

    plt.figure()

    plt.plot(df["episode"], df["reward"], alpha=0.3, label="raw")
    plt.plot(df["episode"], df["reward_smooth"], label="smoothed")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward Curve")
    plt.legend()

    out_path = os.path.join(exp_path, "reward_curve.png")
    os.makedirs(exp_path, exist_ok=True)
    plt.savefig(out_path)

    print(f"Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    args = parser.parse_args()

    plot_experiment(args.path)


if __name__ == "__main__":
    main()

    