class RSIReversalStrategy:
    def __init__(self, window=7, lower=15, upper=45):
        self.window = window
        self.lower = lower
        self.upper = upper

    def compute_rsi(self, series):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(self.window).mean()
        avg_loss = loss.rolling(self.window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_signals(self, df):
        df = df.copy()
        df["RSI"] = self.compute_rsi(df["Close"]).bfill()

        df["signal"] = 0
        df.loc[df["RSI"] < self.lower, "signal"] = 1
        df.loc[df["RSI"] > self.upper, "signal"] = -1

        return df
