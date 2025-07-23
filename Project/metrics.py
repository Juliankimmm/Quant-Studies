def calculate_metrics(trades, equity_curve, regime):
    from numpy import std, mean
    import numpy as np

    returns = equity_curve.pct_change().dropna()
    sharpe = mean(returns) / std(returns) * (252**0.5)
    sortino = mean(returns) / std(returns[returns < 0]) * (252**0.5)
    cagr = (equity_curve.iloc[-1]) ** (1 / (len(equity_curve) / 252)) - 1
    drawdown = equity_curve / equity_curve.cummax() - 1
    calmar = cagr / abs(drawdown.min())

    win_rate = trades["wins"] / trades["total"] if trades["total"] > 0 else 0
    reward_risk = trades["avg_win"] / abs(trades["avg_loss"]) if trades["avg_loss"] != 0 else float("inf")
    profit_factor = trades["gross_profit"] / abs(trades["gross_loss"]) if trades["gross_loss"] != 0 else float("inf")

    return {
        "Market Regime": regime,
        "Sharpe Ratio": round(sharpe, 2),
        "Sortino Ratio": round(sortino, 2),
        "Calmar Ratio": round(calmar, 2),
        "CAGR": round(cagr * 100, 2),
        "Max Drawdown": f"{round(drawdown.min() * 100, 2)}%",
        "Profit Factor": round(profit_factor, 2),
        "Win Rate": f"{round(win_rate * 100, 2)}%",
        "Trade Count": trades["total"],
        "Average Win": round(trades["avg_win"], 4),
        "Average Loss": round(trades["avg_loss"], 4),
        "Reward-to-Risk Ratio": round(reward_risk, 2),
        "Average Trade Duration (days)": trades["avg_duration"],
        "Average Trade Return": round(trades["avg_return"] * 100, 2)
    }
