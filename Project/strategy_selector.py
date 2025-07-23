from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.breakout import BreakoutStrategy
from strategies.ma_crossover import MACrossoverStrategy
from strategies.rsi_reversal import RSIReversalStrategy
from strategies.bollinger_band import BollingerBandStrategy
from strategies.ensemble import EnsembleStrategy
from strategies.ml_strategy import MLStrategy

def choose_strategy(ticker, regime):
    print(f"Selecting strategy for Ticker: {ticker}, Market Regime: {regime}")

    if "BTC" in ticker.upper():
        print("Strategy Chosen: Ensemble (Momentum + BollingerBand)")
        return EnsembleStrategy([
            MomentumStrategy(threshold=0.002),
            BollingerBandStrategy(window=21, num_std=2.5)
        ])
    
    elif regime == "bull":
        print("Strategy Chosen: Ensemble (MACrossover + Momentum)")
        return EnsembleStrategy([
            MACrossoverStrategy(short_window=15, long_window=40),
            MomentumStrategy(threshold=0.001)
        ])
    
    elif regime == "bear":
        print("Strategy Chosen: Ensemble (RSIReversal + BollingerBand)")
        return EnsembleStrategy([
            RSIReversalStrategy(window=7, lower=15, upper=45),
            BollingerBandStrategy(window=10, num_std=3.0)
        ])
    
    elif regime == "sideways" or ticker.upper() in ['AAPL', 'TSLA']:
        print("Strategy Chosen: MLStrategy (Random Forest)")
        return MLStrategy(n_estimators=200)
    
    else:
        print("Strategy Chosen: Ensemble (Breakout + Mean Reversion)")
        return EnsembleStrategy([
            BreakoutStrategy(window=15),
            MeanReversionStrategy(window=18, z_entry=1.2)
        ])
