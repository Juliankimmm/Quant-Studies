import pandas as pd

def backtest(signals, initial_capital=10000):
    """
    Simple backtester: simulates trading based on signals.
    signals: pd.DataFrame with 'signal' column: 1 for buy, -1 for sell, 0 for hold.
    Assumes entering full position on buy, fully out on sell.
    initial_capital: starting portfolio value.

    Returns:
      equity_curve: pd.Series with portfolio value over time
      trades: list of dicts with trade details (entry_date, exit_date, pnl)
    """
    df = signals.copy()
    df['position'] = 0

    # Generate positions based on signals
    # Assume positions switch immediately on signal
    df['position'] = df['signal'].replace(to_replace=0, method='ffill').fillna(0)

    # Calculate daily returns of the underlying asset
    df['returns'] = df['Close'].pct_change().fillna(0)

    # Portfolio returns = position * daily returns
    df['strategy_returns'] = df['position'].shift(1) * df['returns']

    # Equity curve
    df['equity'] = (1 + df['strategy_returns']).cumprod() * initial_capital

    # Extract trades
    trades = []
    in_position = False
    entry_price = None
    entry_date = None

    for i in range(1, len(df)):
        prev_pos = df['position'].iloc[i-1]
        curr_pos = df['position'].iloc[i]
        price = df['Close'].iloc[i]
        date = df.index[i]

        # Enter position
        if prev_pos == 0 and curr_pos != 0:
            entry_price = price
            entry_date = date
            in_position = True

        # Exit position
        elif prev_pos != 0 and curr_pos == 0 and in_position:
            exit_price = price
            exit_date = date
            pnl = (exit_price - entry_price) / entry_price * 100 * prev_pos  # % return times direction
            trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'pnl_pct': pnl
            })
            in_position = False

    return df['equity'], trades
