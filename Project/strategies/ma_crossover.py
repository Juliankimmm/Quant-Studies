class MACrossoverStrategy:
    def __init__(self, short_window=20, long_window=50):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, df):
        df = df.copy()
        df["signal"] = 0
        df["short_ma"] = df["Close"].rolling(self.short_window).mean()
        df["long_ma"] = df["Close"].rolling(self.long_window).mean()
        df.loc[df["short_ma"] > df["long_ma"], "signal"] = 1
        df.loc[df["short_ma"] < df["long_ma"], "signal"] = -1
        return df