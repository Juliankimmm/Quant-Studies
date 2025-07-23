# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from datetime import datetime
from sklearn.mixture import GaussianMixture
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Load historical stock data for AAPL
ticker = 'AAPL'
start = '2018-01-01'
end = datetime.today().strftime('%Y-%m-%d')

df = yf.download(ticker, start=start, end=end, auto_adjust=True)
df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

# Flatten MultiIndex columns if present
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Add technical indicators for regime detection and trading
def add_indicators(df, rsi_window=14, ma_short=20, ma_long=50, bb_window=20):
    """Add technical indicators used for regime detection and trading"""
    df = df.copy()
    close = df['Close']
    
    # Price-based indicators
    df['returns'] = close.pct_change()
    df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
    df['ma_short'] = ta.trend.SMAIndicator(close, window=ma_short).sma_indicator()
    df['ma_long'] = ta.trend.SMAIndicator(close, window=ma_long).sma_indicator()
    df['ma_ratio'] = df['ma_short'] / df['ma_long']
    
    # Momentum indicators
    df['rsi'] = ta.momentum.RSIIndicator(close, window=rsi_window).rsi()
    df['macd'] = ta.trend.MACD(close).macd_diff()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close, window=bb_window)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Volume indicators
    df['volume_sma'] = df['Volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_sma']
    
    return df.dropna()

class RegimeSwitchingStrategy:
    def __init__(self, n_regimes=3):
        self.n_regimes = n_regimes
        self.regime_model = None
        self.regimes = None
        self.regime_names = ['Bear Market', 'Sideways Market', 'Bull Market']
        
    def detect_regimes(self, df, features=['returns', 'volatility', 'ma_ratio', 'bb_width']):
        """Detect market regimes using Gaussian Mixture Model"""
        
        # Prepare feature matrix
        feature_data = df[features].dropna()
        X = feature_data.values
        
        # Standardize features
        X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        
        # Fit Gaussian Mixture Model
        self.regime_model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type='full',
            random_state=42,
            max_iter=200
        )
        
        regime_labels = self.regime_model.fit_predict(X_std)
        
        # Map regimes based on average returns (ascending order)
        regime_returns = []
        for i in range(self.n_regimes):
            mask = regime_labels == i
            avg_return = feature_data.loc[mask, 'returns'].mean()
            regime_returns.append((i, avg_return))
        
        # Sort by average returns and create mapping
        regime_returns.sort(key=lambda x: x[1])
        regime_mapping = {old: new for new, (old, _) in enumerate(regime_returns)}
        
        # Apply mapping
        mapped_regimes = np.array([regime_mapping[label] for label in regime_labels])
        
        # Add regimes to dataframe
        regime_df = df.loc[feature_data.index].copy()
        regime_df['regime'] = mapped_regimes
        
        self.regimes = regime_df['regime']
        return regime_df
    
    def analyze_regimes(self, df):
        """Analyze characteristics of each regime"""
        regime_stats = {}
        
        for regime in range(self.n_regimes):
            mask = df['regime'] == regime
            regime_data = df[mask]
            
            stats = {
                'count': len(regime_data),
                'avg_return': regime_data['returns'].mean() * 252,  # Annualized
                'volatility': regime_data['returns'].std() * np.sqrt(252),  # Annualized
                'avg_rsi': regime_data['rsi'].mean(),
                'avg_bb_width': regime_data['bb_width'].mean(),
                'avg_volume_ratio': regime_data['volume_ratio'].mean()
            }
            
            regime_stats[regime] = stats
        
        return regime_stats
    
    def get_trading_signal(self, row, regime, rsi_params, bb_params):
        """Generate trading signals based on regime and technical indicators"""
        
        if regime == 0:  # Bear Market - Conservative, short bias
            rsi_buy, rsi_sell = rsi_params['bear']
            bb_buy, bb_sell = bb_params['bear']
            
            # More restrictive buying in bear markets
            if row['rsi'] < rsi_buy and row['bb_position'] < bb_buy:
                return 'buy'
            elif row['rsi'] > rsi_sell or row['bb_position'] > bb_sell:
                return 'sell'
                
        elif regime == 1:  # Sideways Market - Mean reversion
            rsi_buy, rsi_sell = rsi_params['sideways']
            bb_buy, bb_sell = bb_params['sideways']
            
            # Mean reversion strategy
            if row['rsi'] < rsi_buy and row['bb_position'] < bb_buy:
                return 'buy'
            elif row['rsi'] > rsi_sell and row['bb_position'] > bb_sell:
                return 'sell'
                
        elif regime == 2:  # Bull Market - Trend following
            rsi_buy, rsi_sell = rsi_params['bull']
            bb_buy, bb_sell = bb_params['bull']
            
            # More aggressive buying in bull markets
            if row['rsi'] < rsi_buy and row['ma_ratio'] > 1.01:  # Uptrend confirmation
                return 'buy'
            elif row['rsi'] > rsi_sell or row['ma_ratio'] < 0.99:
                return 'sell'
        
        return 'hold'
    
    def backtest(self, df, rsi_params, bb_params):
        """Backtest the regime-switching strategy"""
        
        capital = 10000.0
        position = 0.0
        cash = capital
        equity = [capital]
        trades = []
        
        for i in range(1, len(df)):
            current_row = df.iloc[i]
            regime = int(current_row['regime'])
            price = float(current_row['Close'])
            
            signal = self.get_trading_signal(current_row, regime, rsi_params, bb_params)
            
            if signal == 'buy' and cash > 0:
                position = cash / price
                cash = 0.0
                trades.append(('buy', df.index[i], price, regime))
                
            elif signal == 'sell' and position > 0:
                cash = position * price
                position = 0.0
                trades.append(('sell', df.index[i], price, regime))
            
            total_value = cash + position * price
            equity.append(total_value)
        
        return equity[1:], trades  # Remove initial capital from equity curve

def optimize_regime_strategy():
    """Optimize parameters for regime-switching strategy"""
    
    # Add indicators to dataframe
    df_with_indicators = add_indicators(df)
    
    # Initialize strategy
    strategy = RegimeSwitchingStrategy(n_regimes=3)
    
    # Detect regimes
    regime_df = strategy.detect_regimes(df_with_indicators)
    
    def objective(bear_rsi_buy, bear_rsi_sell, sideways_rsi_buy, sideways_rsi_sell, 
                  bull_rsi_buy, bull_rsi_sell, bear_bb_buy, bear_bb_sell,
                  sideways_bb_buy, sideways_bb_sell, bull_bb_buy, bull_bb_sell):
        
        rsi_params = {
            'bear': (bear_rsi_buy, bear_rsi_sell),
            'sideways': (sideways_rsi_buy, sideways_rsi_sell),
            'bull': (bull_rsi_buy, bull_rsi_sell)
        }
        
        bb_params = {
            'bear': (bear_bb_buy, bear_bb_sell),
            'sideways': (sideways_bb_buy, sideways_bb_sell),
            'bull': (bull_bb_buy, bull_bb_sell)
        }
        
        # Validate parameters
        for regime_params in rsi_params.values():
            if regime_params[0] >= regime_params[1]:
                return -1000
        
        for regime_params in bb_params.values():
            if regime_params[0] >= regime_params[1]:
                return -1000
        
        try:
            equity_curve, _ = strategy.backtest(regime_df, rsi_params, bb_params)
            
            if len(equity_curve) < 2:
                return -1000
            
            returns = pd.Series(equity_curve).pct_change().dropna()
            
            if returns.empty or returns.std() == 0:
                return -1000
            
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
            
            if pd.isna(sharpe) or not np.isfinite(sharpe):
                return -1000
                
            return sharpe
            
        except Exception as e:
            return -1000
    
    # Parameter bounds for optimization
    pbounds = {
        # RSI parameters for each regime
        'bear_rsi_buy': (20, 35),
        'bear_rsi_sell': (65, 85),
        'sideways_rsi_buy': (25, 40),
        'sideways_rsi_sell': (60, 80),
        'bull_rsi_buy': (30, 45),
        'bull_rsi_sell': (55, 75),
        
        # Bollinger Band position parameters
        'bear_bb_buy': (0.1, 0.3),
        'bear_bb_sell': (0.7, 0.9),
        'sideways_bb_buy': (0.1, 0.3),
        'sideways_bb_sell': (0.7, 0.9),
        'bull_bb_buy': (0.2, 0.4),
        'bull_bb_sell': (0.6, 0.8),
    }
    
    # Setup and run optimizer
    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )
    
    optimizer.maximize(init_points=10, n_iter=25)
    
    return optimizer.max, strategy, regime_df

# Run optimization
print("=== Running Regime-Switching Strategy Optimization ===")
best_result, strategy, regime_df = optimize_regime_strategy()

# Extract best parameters
best_params = best_result['params']
rsi_params = {
    'bear': (best_params['bear_rsi_buy'], best_params['bear_rsi_sell']),
    'sideways': (best_params['sideways_rsi_buy'], best_params['sideways_rsi_sell']),
    'bull': (best_params['bull_rsi_buy'], best_params['bull_rsi_sell'])
}

bb_params = {
    'bear': (best_params['bear_bb_buy'], best_params['bear_bb_sell']),
    'sideways': (best_params['sideways_bb_buy'], best_params['sideways_bb_sell']),
    'bull': (best_params['bull_bb_buy'], best_params['bull_bb_sell'])
}

print(f"\n=== Optimization Results ===")
print(f"Best Sharpe Ratio: {best_result['target']:.4f}")
print("\nOptimal RSI Parameters:")
for regime, params in rsi_params.items():
    print(f"  {regime.capitalize()}: Buy < {params[0]:.2f}, Sell > {params[1]:.2f}")

print("\nOptimal Bollinger Band Parameters:")
for regime, params in bb_params.items():
    print(f"  {regime.capitalize()}: Buy < {params[0]:.2f}, Sell > {params[1]:.2f}")

# Analyze regime characteristics
regime_stats = strategy.analyze_regimes(regime_df)
print(f"\n=== Regime Analysis ===")
for regime, stats in regime_stats.items():
    regime_name = strategy.regime_names[regime]
    print(f"\n{regime_name} (Regime {regime}):")
    print(f"  Occurrences: {stats['count']} days ({stats['count']/len(regime_df)*100:.1f}%)")
    print(f"  Avg Annual Return: {stats['avg_return']*100:.2f}%")
    print(f"  Avg Annual Volatility: {stats['volatility']*100:.2f}%")
    print(f"  Avg RSI: {stats['avg_rsi']:.1f}")
    print(f"  Avg BB Width: {stats['avg_bb_width']:.3f}")

# Final backtest with optimal parameters
equity_curve, trades = strategy.backtest(regime_df, rsi_params, bb_params)

# Buy and Hold comparison
initial_cash = 10000.0
buy_price = float(regime_df['Close'].iloc[0])
shares_bought = initial_cash / buy_price
buy_and_hold_equity = shares_bought * regime_df['Close'].values[1:]

# Performance metrics calculation
def compute_metrics(equity_curve, dates):
    equity = pd.Series(equity_curve, index=dates)
    returns = equity.pct_change().dropna()
    
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    annualized_return = (1 + total_return) ** (252 / len(equity)) - 1
    annualized_volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else np.nan
    
    # Drawdown calculation
    cumulative_max = equity.cummax()
    drawdown = (equity - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()
    
    return {
        'Final Portfolio Value': equity.iloc[-1],
        'Total Return (%)': total_return * 100,
        'Annualized Return (%)': annualized_return * 100,
        'Annualized Volatility (%)': annualized_volatility * 100,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown (%)': max_drawdown * 100,
    }

# Calculate performance metrics
regime_metrics = compute_metrics(equity_curve, regime_df.index[1:])
buy_hold_metrics = compute_metrics(buy_and_hold_equity, regime_df.index[1:])

print(f"\n=== Performance Comparison ===")
print(f"{'Metric':<25} | {'Regime Strategy':>17} | {'Buy and Hold':>15}")
print("-" * 62)
for metric in regime_metrics.keys():
    regime_val = regime_metrics[metric]
    buy_hold_val = buy_hold_metrics[metric]
    if 'Ratio' in metric:
        print(f"{metric:<25} | {regime_val:17.4f} | {buy_hold_val:15.4f}")
    else:
        print(f"{metric:<25} | {regime_val:17,.2f} | {buy_hold_val:15,.2f}")

# Trading activity analysis
print(f"\n=== Trading Activity ===")
print(f"Total trades: {len(trades)}")
regime_trades = {0: 0, 1: 0, 2: 0}
for trade in trades:
    regime_trades[trade[3]] += 1

for regime, count in regime_trades.items():
    regime_name = strategy.regime_names[regime]
    print(f"Trades in {regime_name}: {count}")

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Equity curves comparison
axes[0, 0].plot(regime_df.index[1:], equity_curve, label='Regime-Switching Strategy', linewidth=2)
axes[0, 0].plot(regime_df.index[1:], buy_and_hold_equity, label='Buy and Hold', linewidth=2)
axes[0, 0].set_title('Portfolio Value Comparison')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Portfolio Value ($)')
axes[0, 0].legend()
axes[0, 0].grid(True)

# 2. Regime timeline
colors = ['red', 'orange', 'green']
regime_colors = [colors[r] for r in regime_df['regime']]
axes[0, 1].scatter(regime_df.index, regime_df['Close'], c=regime_colors, alpha=0.6, s=1)
axes[0, 1].set_title('Market Regimes Over Time')
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Price ($)')
axes[0, 1].legend(['Bear', 'Sideways', 'Bull'], loc='upper left')

# 3. RSI with regime-specific thresholds
axes[1, 0].plot(regime_df.index, regime_df['rsi'], label='RSI', color='purple', alpha=0.7)
axes[1, 0].axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold')
axes[1, 0].axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought')
axes[1, 0].set_title('RSI Over Time')
axes[1, 0].set_xlabel('Date')
axes[1, 0].set_ylabel('RSI')
axes[1, 0].legend()
axes[1, 0].grid(True)

# 4. Drawdown comparison
regime_equity = pd.Series(equity_curve, index=regime_df.index[1:])
regime_cummax = regime_equity.cummax()
regime_drawdown = (regime_equity - regime_cummax) / regime_cummax * 100

buy_hold_equity_series = pd.Series(buy_and_hold_equity, index=regime_df.index[1:])
buy_hold_cummax = buy_hold_equity_series.cummax()
buy_hold_drawdown = (buy_hold_equity_series - buy_hold_cummax) / buy_hold_cummax * 100

axes[1, 1].fill_between(regime_df.index[1:], regime_drawdown, 0, alpha=0.3, label='Regime Strategy')
axes[1, 1].fill_between(regime_df.index[1:], buy_hold_drawdown, 0, alpha=0.3, label='Buy and Hold')
axes[1, 1].set_title('Drawdown Comparison')
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Drawdown (%)')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

print(f"\n=== Strategy Summary ===")
print(f"The regime-switching strategy automatically adapts to different market conditions:")
print(f"• Bear markets: Conservative approach with strict entry/exit rules")
print(f"• Sideways markets: Mean-reversion strategy using RSI and Bollinger Bands")  
print(f"• Bull markets: Trend-following approach with momentum confirmation")
print(f"This adaptive approach achieved a Sharpe ratio of {best_result['target']:.4f}")