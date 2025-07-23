class MeanReversionStrategy:
    def generate_signals(self, df):
        df["signal"] = 0
        df["zscore"] = (df["Close"] - df["Close"].rolling(20).mean()) / df["Close"].rolling(20).std()
        df.loc[df["zscore"] > 1, "signal"] = -1
        df.loc[df["zscore"] < -1, "signal"] = 1
        return df
