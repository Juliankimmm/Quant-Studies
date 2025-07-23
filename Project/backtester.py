import pandas as pd

def backtest(df):
    df["daily_return"] = df["Close"].pct_change()
    df["strategy_return"] = df["signal"].shift(1) * df["daily_return"]
    df["equity_curve"] = (1 + df["strategy_return"]).cumprod()

    trades = {
        "wins": len(df[df["strategy_return"] > 0]),
        "losses": len(df[df["strategy_return"] < 0]),
        "total": len(df[df["signal"] != 0]),
        "gross_profit": df[df["strategy_return"] > 0]["strategy_return"].sum(),
        "gross_loss": df[df["strategy_return"] < 0]["strategy_return"].sum(),
        "avg_win": df[df["strategy_return"] > 0]["strategy_return"].mean(),
        "avg_loss": df[df["strategy_return"] < 0]["strategy_return"].mean(),
        "avg_duration": 1,  # assume 1-day hold, can improve
        "avg_return": df[df["signal"] != 0]["strategy_return"].mean()
    }

    return df["equity_curve"], trades
