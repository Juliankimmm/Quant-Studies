import pandas as pd
from sklearn.ensemble import RandomForestClassifier

class MLStrategy:
    def __init__(self, train_ratio=0.7, n_estimators=100):
        self.train_ratio = train_ratio
        self.n_estimators = n_estimators
        self.model = RandomForestClassifier(n_estimators=self.n_estimators)
        self.is_fitted = False

    def _prepare_features(self, df):
        df = df.copy()
        df['return'] = df['Close'].pct_change()
        df['target'] = (df['return'].shift(-1) > 0).astype(int)  # Next day up/down

        # Simple features - can add more here
        df['ma_10'] = df['Close'].rolling(10).mean()
        df['ma_50'] = df['Close'].rolling(50).mean()
        df['momentum'] = df['Close'] - df['Close'].shift(10)
        df = df.dropna()

        features = ['ma_10', 'ma_50', 'momentum']
        return df, features

    def generate_signals(self, df):
        df, features = self._prepare_features(df)
        split = int(len(df) * self.train_ratio)

        train_df = df.iloc[:split]
        test_df = df.iloc[split:]

        X_train = train_df[features]
        y_train = train_df['target']

        X_test = test_df[features]
        test_index = test_df.index

        # Fit model only once
        if not self.is_fitted:
            self.model.fit(X_train, y_train)
            self.is_fitted = True

        preds = self.model.predict(X_test)

        signals = pd.Series(0, index=df.index)
        # Assign signals only on test set period
        # +1 if model predicts price up next day, -1 if down
        signals.loc[test_index] = [1 if p == 1 else -1 for p in preds]

        df['signal'] = signals
        return df[['signal']]

