import yfinance as yf
import pandas as pd
import numpy as np

# -------------------------------
# Core Functions
# -------------------------------

def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df = df[['Close']]
    df['Returns'] = df['Close'].pct_change()
    return df.dropna()

def generate_signals(df, fast, slow):
    df['SMA_fast'] = df['Close'].rolling(fast).mean()
    df['SMA_slow'] = df['Close'].rolling(slow).mean()
    df['Signal'] = 0
    df.loc[df['SMA_fast'] > df['SMA_slow'], 'Signal'] = 1
    df['Position'] = df['Signal'].shift(1)
    return df.dropna()

def backtest(df, initial_capital=10000):
    df['Strategy_Returns'] = df['Position'] * df['Returns']
    df['Equity'] = (1 + df['Strategy_Returns']).cumprod() * initial_capital
    return df

def performance_metrics(df):
    if df['Strategy_Returns'].std() == 0:
        return 0, 0, 0  # avoid division errors
    cagr = (df['Equity'].iloc[-1] / df['Equity'].iloc[0]) ** (1 / (len(df)/252)) - 1
    sharpe = df['Strategy_Returns'].mean() / df['Strategy_Returns'].std() * np.sqrt(252)
    running_max = df['Equity'].cummax()
    drawdown = df['Equity'] / running_max - 1
    max_dd = drawdown.min()
    return cagr, sharpe, max_dd

# -------------------------------
# Grid Search for Optimization
# -------------------------------

def tune_parameters(ticker, fast_range, slow_range):
    print(f"\n Testing SMA ranges on {ticker}")
    df_raw = load_data(ticker, "2018-01-01", "2024-01-01")
    
    results = []

    for fast in fast_range:
        for slow in slow_range:
            if fast >= slow:
                continue  # invalid combo
            df = df_raw.copy()
            df = generate_signals(df, fast, slow)
            df = backtest(df)
            cagr, sharpe, max_dd = performance_metrics(df)
            results.append({
                'Ticker': ticker,
                'Fast': fast,
                'Slow': slow,
                'CAGR': round(cagr * 100, 2),
                'Sharpe': round(sharpe, 2),
                'Max Drawdown': round(max_dd * 100, 2),
                'Final Equity': round(df['Equity'].iloc[-1], 2)
            })
    
    return pd.DataFrame(results).sort_values(by='Sharpe', ascending=False)

# -------------------------------
# MAIN
# -------------------------------

if __name__ == "__main__":
    fast_range = range(5, 30, 5)
    slow_range = range(40, 120, 10)
    tickers = ['AAPL', 'TSLA', 'MSFT']

    for ticker in tickers:
        df_result = tune_parameters(ticker, fast_range, slow_range)
        print(df_result.head(5))  # top 5 best settings
