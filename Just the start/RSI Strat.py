import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from bayes_opt import BayesianOptimization
import ta

# === Step 1: Load Historical Data ===
ticker = 'AAPL'
start = '2018-01-01'
end = datetime.today().strftime('%Y-%m-%d')

df = yf.download(ticker, start=start, end=end, auto_adjust=True)
df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

# === Step 2: Technical Indicator Calculation ===
def apply_indicators(data, window=14):
    data = data.copy()
    close = data['Close']
    if isinstance(close, pd.DataFrame):
        close = close.squeeze()
    data['RSI'] = ta.momentum.RSIIndicator(close, window=window).rsi()
    data['SMA'] = ta.trend.SMAIndicator(close, window=window).sma_indicator()
    return data.dropna()

# === Step 3: Backtest Logic ===
def backtest(data, rsi_buy, rsi_sell):
    capital = 10000.0
    cash, position = capital, 0
    equity_curve = []

    for i in range(1, len(data)):
        price = float(data['Close'].iat[i])
        rsi = float(data['RSI'].iat[i])

        if rsi < rsi_buy and cash > 0:
            position = cash / price
            cash = 0
            # print(f"Buy at index {i}, price={price:.2f}, RSI={rsi:.2f}")
        elif rsi > rsi_sell and position > 0:
            cash = position * price
            position = 0
            # print(f"Sell at index {i}, price={price:.2f}, RSI={rsi:.2f}")

        equity_curve.append(cash + position * price)

    returns = pd.Series(equity_curve).pct_change().dropna()
    if returns.empty or returns.std() == 0:
        return -1000

    sharpe = returns.mean() / returns.std() * np.sqrt(252)
    return sharpe if np.isfinite(sharpe) else -1000

# === Step 4: Optimization ===
def optimize():
    def objective(rsi_buy, rsi_sell, window):
        rsi_buy, rsi_sell, window = float(rsi_buy), float(rsi_sell), int(round(window))
        if rsi_buy >= rsi_sell:
            return -100
        try:
            temp = apply_indicators(df, window)
            return backtest(temp, rsi_buy, rsi_sell)
        except Exception:
            return -100

    pbounds = {'rsi_buy': (10, 40), 'rsi_sell': (60, 90), 'window': (5, 30)}
    optimizer = BayesianOptimization(f=objective, pbounds=pbounds, random_state=42, verbose=0)
    optimizer.maximize(init_points=5, n_iter=15)
    return optimizer.max

best = optimize()

# === Step 5: Final Backtest ===
best_rsi_buy = best['params']['rsi_buy']
best_rsi_sell = best['params']['rsi_sell']
best_window = int(round(best['params']['window']))

df_final = apply_indicators(df, best_window)

def backtest_equity(data, rsi_buy, rsi_sell):
    capital = 10000.0
    cash, position = capital, 0
    equity_curve = []

    close_series = data['Close']
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.squeeze()  # Ensure Series, not DataFrame

    for i in range(1, len(data)):
        price = float(close_series.iat[i])
        rsi = float(data['RSI'].iat[i])

        if rsi < rsi_buy and cash > 0:
            position = cash / price
            cash = 0
        elif rsi > rsi_sell and position > 0:
            cash = position * price
            position = 0

        equity_curve.append(cash + position * price)

    return equity_curve


rsi_equity = backtest_equity(df_final, best_rsi_buy, best_rsi_sell)

# === Step 6: Buy and Hold ===
initial_cash = 10000.0
close_series = df_final['Close']
if isinstance(close_series, pd.DataFrame):
    close_series = close_series.squeeze()

buy_price = close_series.iat[0]
shares = initial_cash / buy_price
# Buy & Hold equity from day 1 onwards, must be 1D array to match rsi_equity length
buy_hold_equity = (shares * df_final['Close'].iloc[1:]).to_numpy().flatten()

# === Step 7: Metrics ===
def compute_metrics(equity, index):
    if not isinstance(equity, pd.Series):
        equity = pd.Series(equity, index=index[1:])

    returns = equity.pct_change().dropna()
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    annual_ret = (1 + total_return) ** (252 / len(equity)) - 1
    vol = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else np.nan
    drawdown = (equity - equity.cummax()) / equity.cummax()
    max_drawdown = drawdown.min()

    return {
        'Final Portfolio Value': equity.iloc[-1],
        'Total Return (%)': total_return * 100,
        'Annualized Return (%)': annual_ret * 100,
        'Annualized Volatility (%)': vol * 100,
        'Sharpe Ratio': sharpe,
        'Max Drawdown (%)': max_drawdown * 100
    }

rsi_metrics = compute_metrics(rsi_equity, df_final.index)
buy_hold_metrics = compute_metrics(buy_hold_equity, df_final.index)

# === Step 8: Print Results ===
print("\n=== Optimization Complete ===")
print(f"Best Sharpe Ratio: {best['target']:.4f}")
print(f"RSI Buy Threshold:  {best_rsi_buy:.2f}")
print(f"RSI Sell Threshold: {best_rsi_sell:.2f}")
print(f"RSI Window:         {best_window}\n")

print("=== Performance Metrics Comparison ===")
print(f"{'Metric':<25} | {'RSI Strategy':>15} | {'Buy and Hold':>15}")
print("-" * 60)
for key in rsi_metrics:
    rsi_val = rsi_metrics[key]
    bh_val = buy_hold_metrics[key]
    if isinstance(rsi_val, float) and 'Ratio' in key:
        print(f"{key:<25} | {rsi_val:15.4f} | {bh_val:15.4f}")
    else:
        print(f"{key:<25} | {rsi_val:15,.2f} | {bh_val:15,.2f}")

# === Step 9: Plotting ===
plt.figure(figsize=(12, 6))
plt.plot(df_final.index[1:], rsi_equity, label='RSI Strategy')
plt.plot(df_final.index[1:], buy_hold_equity, label='Buy and Hold')
plt.title('Equity Curve Comparison')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()