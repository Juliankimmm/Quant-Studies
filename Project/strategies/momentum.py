class MomentumStrategy:
    def __init__(self, threshold=0):
        self.threshold = threshold

    def generate_signals(self, df):
        df = df.copy()
        df["signal"] = 0
        df["return"] = df["Close"].pct_change()
        df["momentum"] = df["return"].rolling(10).mean()
        df.loc[df["momentum"] > self.threshold, "signal"] = 1
        df.loc[df["momentum"] < -self.threshold, "signal"] = -1
        return df
