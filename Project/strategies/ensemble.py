class EnsembleStrategy:
    def __init__(self, strategies):
        self.strategies = strategies

    def generate_signals(self, df):
        df = df.copy()
        signal_sum = 0
        for strategy in self.strategies:
            df_temp = strategy.generate_signals(df)
            signal_sum += df_temp["signal"]
        df["signal"] = signal_sum.round().clip(-1, 1)
        return df
