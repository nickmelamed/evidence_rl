def smooth_reward(df, window: int = 5):
    """Rolling-average smoothed reward column, added in place. Returns df."""
    df["reward_smooth"] = df["reward"].rolling(window=window, min_periods=1).mean()
    return df
