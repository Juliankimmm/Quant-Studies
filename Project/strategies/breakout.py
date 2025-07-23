class BreakoutStrategy:
    def generate_signals(self, df):
        df["signal"] = 0
        df["high_break"] = df["High"].rolling(20).max()
        df["low_break"] = df["Low"].rolling(20).min()
        df.loc[df["Close"] > df["high_break"], "signal"] = 1
        df.loc[df["Close"] < df["low_break"], "signal"] = -1
        return df
