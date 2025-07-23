class BollingerBandStrategy:
    def __init__(self, window=10, num_std=3):
        self.window = window
        self.num_std = num_std

    def generate_signals(self, df):
        df = df.copy()
        df["ma"] = df["Close"].rolling(self.window).mean().bfill()
        df["std"] = df["Close"].rolling(self.window).std().bfill()
        df["upper"] = df["ma"] + self.num_std * df["std"]
        df["lower"] = df["ma"] - self.num_std * df["std"]

        df["signal"] = 0

        close = df["Close"].squeeze()
        lower = df["lower"].squeeze()
        close, lower = close.align(lower, join="inner", axis=0)
        df.loc[close.index, "signal"] = (close < lower).astype(int)

        upper = df["upper"].squeeze()
        close2 = df["Close"].squeeze()
        close2, upper = close2.align(upper, join="inner", axis=0)
        # Where price > upper band, signal = -1
        df.loc[close2.index, "signal"] = df.loc[close2.index, "signal"].where(~(close2 > upper), -1)

        return df

