import pandas as pd

def detect_regime(price_series):
    # Squeeze to convert shape (n,1) to (n,)
    price_series = pd.Series(price_series.squeeze()).dropna().astype(float)

    if len(price_series) < 200:
        return "unknown"

    fast_ma = price_series.rolling(window=50).mean()
    slow_ma = price_series.rolling(window=200).mean()

    fast_val = fast_ma.iloc[-1]
    slow_val = slow_ma.iloc[-1]

    if pd.isna(fast_val) or pd.isna(slow_val):
        return "unknown"

    if fast_val > slow_val:
        return "bull"
    elif fast_val < slow_val:
        return "bear"
    else:
        return "sideways"
