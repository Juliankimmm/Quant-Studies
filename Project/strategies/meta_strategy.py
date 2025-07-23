from strategies.bollinger_band import BollingerBandStrategy
from strategies.rsi_reversal import RSIReversalStrategy
from strategies.ml_strategy import MLStrategy
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.breakout import BreakoutStrategy
from strategies.ma_crossover import MACrossoverStrategy

from backtest import backtest

class MetaStrategyRunner:
    def __init__(self, data, ticker):
        self.data = data
        self.ticker = ticker
        self.strategies = {
            "BollingerBand": BollingerBandStrategy(),
            "RSIReversal": RSIReversalStrategy(),
            "ML": MLStrategy(),
            "Momentum": MomentumStrategy(),
            "MeanReversion": MeanReversionStrategy(),
            "Breakout": BreakoutStrategy(),
            "MACrossover": MACrossoverStrategy(),
        }

    def evaluate_strategies(self):
        results = []
        for name, strategy in self.strategies.items():
            metrics, _ = backtest(self.data.copy(), strategy)
            results.append((name, strategy, metrics))
        return sorted(results, key=lambda x: x[2]['Sharpe'], reverse=True)  # sort by Sharpe

    def select_and_optimize(self):
        evaluated = self.evaluate_strategies()
        best_name, best_strategy, best_metrics = evaluated[0]
        print(f"\nBest Strategy: {best_name} with Sharpe: {best_metrics['Sharpe']:.2f}")

        # Optional: optimize parameters here (for example if it's ML or BB)
        if hasattr(best_strategy, 'optimize'):
            best_strategy.optimize(self.data)

        return best_strategy
