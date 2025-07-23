import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from data_loader import get_data
from market_regime import detect_regime
from strategy_selector import choose_strategy
from backtester import backtest
from metrics import calculate_metrics
from strategies.meta_strategy import MetaStrategyRunner

def plot_regime_shading(ax, regime_series):
    colors = {'bull': 'green', 'bear': 'red', 'sideways': 'yellow'}
    prev = None
    start = None
    for date, regime in regime_series.items():
        if regime != prev:
            if prev is not None:
                ax.axvspan(start, date, alpha=0.1, color=colors.get(prev, 'grey'))
            start = date
            prev = regime
    ax.axvspan(start, regime_series.index[-1], alpha=0.1, color=colors.get(prev, 'grey'))

def plot_signals(ax, data, signals):
    buys = signals[signals['signal'] == 1]
    sells = signals[signals['signal'] == -1]
    ax.scatter(buys.index, data.loc[buys.index, 'Close'], marker='^', color='green', label='Buy Signal', s=100)
    ax.scatter(sells.index, data.loc[sells.index, 'Close'], marker='v', color='red', label='Sell Signal', s=100)

def plot_equity_curve(ax, equity_curve):
    ax.plot(equity_curve.index, equity_curve.values, label='Equity Curve', color='blue')
    ax.set_ylabel("Portfolio Value")
    ax.legend()

def plot_feature_importance(model, features):
    importances = model.feature_importances_
    feat_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    feat_df.sort_values(by='Importance', ascending=False, inplace=True)

    plt.figure(figsize=(6, 4))
    sns.barplot(x='Importance', y='Feature', data=feat_df)
    plt.title("Feature Importances (MLStrategy)")
    plt.tight_layout()
    plt.show()

def main():
    ticker = input("Enter ticker symbol (e.g. AAPL, BTC-USD): ").upper()
    data = get_data(ticker)

    regime = detect_regime(data["Close"])
    regime_series = pd.Series(regime, index=data.index) if isinstance(regime, str) else regime

    strategy = choose_strategy(ticker, regime_series.iloc[-1])
    signals = strategy.generate_signals(data)
    equity_curve, trades = backtest(signals)
    metrics = calculate_metrics(trades, equity_curve, regime_series.iloc[-1])

    print("\n=== Strategy Metrics ===")
    explanations = {
        'Sharpe Ratio': 'Risk-adjusted return (higher is better)',
        'Sortino Ratio': 'Downside risk-adjusted return',
        'Calmar Ratio': 'Return relative to max drawdown',
        'CAGR': 'Compound annual growth rate',
        'Max Drawdown': 'Largest peak-to-trough loss',
        'Profit Factor': 'Gross profit / gross loss',
        'Win Rate': 'Percentage of profitable trades',
        'Trade Count': 'Number of trades',
        'Average Win': 'Average profit on winning trades',
        'Average Loss': 'Average loss on losing trades',
        'Reward-to-Risk Ratio': 'Average win / average loss',
        'Average Trade Duration (days)': 'Mean trade holding period',
        'Average Trade Return': 'Mean return per trade',
        'Regime Analysis': 'Performance broken down by market regime'
    }
    for k, v in metrics.items():
        print(f"{k}: {v} ; {explanations.get(k, '')}")

    sns.set_style("darkgrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    ax1.set_title(f"{ticker} Price Chart with Strategy Signals and Market Regime")
    ax1.plot(data.index, data['Close'], label='Close Price', color='black')
    plot_regime_shading(ax1, regime_series)
    plot_signals(ax1, data, signals)
    ax1.legend()

    ax2.set_title("Equity Curve Over Time")
    plot_equity_curve(ax2, equity_curve)

    plt.xlabel("Date")
    plt.tight_layout()
    plt.show()

    if hasattr(strategy, "model") and hasattr(strategy.model, "feature_importances_"):
        print("\nVisualizing ML Model Feature Importances...")
        plot_feature_importance(strategy.model, ['ma_10', 'ma_50', 'momentum'])

def print_metrics(results):
    ...
    initial_capital = 100
    final_capital = initial_capital * (1 + results['cagr'] / 100) ** results['years']
    print(f"Initial Capital: ${initial_capital:.2f}")
    print(f"Final Capital:   ${final_capital:.2f}")

if __name__ == "__main__":
    main()
