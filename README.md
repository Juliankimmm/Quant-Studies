# Quantitative Finance Projects and Learning Journey

This repository documents my learning journey in **quantitative finance** as a beginner. I've been building a foundation in algorithmic trading by developing backtestable strategies, applying machine learning concepts like Gaussian Mixture Models, using optimization techniques such as Bayesian Optimization, and understanding core financial metrics. This README explains the structure of my code, what each project does, the terms I've encountered, and what I've learned along the way.

---

## What I've Learned So Far

### 1. **Data Acquisition**
- Used the `yfinance` library to download historical stock data.
- Learned to manipulate and clean financial time series using `pandas`.

**Example:**
```python
import yfinance as yf

df = yf.download('AAPL', start='2018-01-01', end='2024-01-01')
```

### 2. **Technical Indicators**
Integrated indicators using the `ta` (technical analysis) library:

- **SMA (Simple Moving Average)**: average of past n prices.
- **RSI (Relative Strength Index)**: momentum indicator to detect overbought/oversold conditions.
- **MACD (Moving Average Convergence Divergence)**: used to identify momentum changes.
- **Bollinger Bands**: show volatility based on price standard deviation.

### 3. **Signal Generation and Backtesting**
- Built logic to generate buy/sell signals based on SMA and RSI.
- Simulated historical strategy performance using a simple backtest engine.
- Compared strategies like SMA crossovers and RSI-based rules to a buy-and-hold baseline.

### 4. **Performance Evaluation**
Learned key metrics to evaluate a strategy:

- **Sharpe Ratio**: risk-adjusted return; higher is better.
- **CAGR (Compound Annual Growth Rate)**: measures annual return over multiple years.
- **Maximum Drawdown**: worst peak-to-trough decline; lower is better.
- **Annualized Volatility**: standard deviation of returns scaled to yearly basis.

### 5. **Optimization**
- Used Bayesian Optimization to find optimal RSI thresholds or moving average window lengths to maximize Sharpe ratio.
- Defined bounds and constraints to avoid invalid strategies (e.g. buy threshold should be lower than sell).

### 6. **Regime Switching Model**
Implemented a Gaussian Mixture Model (GMM) to detect market regimes:

- **Bear Market**
- **Sideways Market**
- **Bull Market**

Created custom strategy logic depending on regime:
- **Bear**: conservative with stricter sell signals.
- **Sideways**: mean reversion strategy.
- **Bull**: trend-following strategy.

Assigned regimes by clustering features like returns, volatility, moving averages, and Bollinger Band width.

---

## Project Descriptions

### Project 1: SMA Crossover Strategy with Grid Search
- **Goal**: Test various combinations of fast and slow moving averages to find best SMA crossover strategy.
- **Method**: Use a brute-force grid search to evaluate strategies across defined ranges.
- **Metrics**: Sharpe Ratio, CAGR, Max Drawdown.

### Project 2: Regime Switching Strategy
- **Goal**: Automatically detect market regimes and adjust strategy accordingly.
- **Model**: Gaussian Mixture Model classifies the data into three regimes based on statistical patterns.
- **Trading Logic**: Customized for each regime using RSI, Bollinger Band position, and MA ratio.
- **Optimization**: Used Bayesian Optimization to fine-tune thresholds for each regime.

### Project 3: RSI-Based Strategy Optimization
- **Goal**: Backtest and optimize a trading strategy based on RSI values.
- **Method**: Buy when RSI is low (oversold), sell when RSI is high (overbought).
- **Optimization**: Used Bayesian Optimization to find best RSI thresholds and window length.

---

## Terminology Cheat Sheet

| Term | Definition |
|------|------------|
| **SMA (Simple Moving Average)** | Mean of the closing prices over a period. Used to smooth price data. |
| **RSI (Relative Strength Index)** | Momentum indicator measuring the speed and change of price movements. Values below 30 are oversold; above 70 are overbought. |
| **MACD** | Difference between 12-day and 26-day EMAs (exponential moving averages). Indicates momentum. |
| **Bollinger Bands** | Volatility bands placed above/below a moving average. |
| **Sharpe Ratio** | Measures risk-adjusted return: (mean return - risk-free rate) / volatility. |
| **Drawdown** | The decline from a peak in the portfolio to a trough. |
| **Backtesting** | Simulating a trading strategy on historical data to evaluate performance. |
| **Gaussian Mixture Model (GMM)** | Probabilistic clustering algorithm to classify data into multiple groups (regimes). |
| **Bayesian Optimization** | Method for finding the max/min of a function by building a probabilistic model of it. |
| **Buy and Hold** | Passive investment strategy: buy once and hold for the entire period. |
| **Volatility** | Standard deviation of returns, indicating risk. |
| **Equity Curve** | Graph of the portfolio value over time. |

---

## File Structure

```
quant-strategies/
│
├── quant_start.py              # Grid Search SMA strategy
├── regime_switching_strat.py          # GMM-based strategy
├── rsi_strat.py             # RSI + Bayesian Optimization
├── README.md                    # Learning summary and documentation
```

---

## Key Takeaways

- Quantitative finance involves a lot of data-driven testing.
- No strategy is universally best — it depends on the market regime.
- Risk-adjusted performance (Sharpe ratio) is more important than raw return.
- Backtesting is not prediction; it only evaluates how a strategy would have performed historically.
- Bayesian Optimization is a smart alternative to brute-force tuning of parameters.
- Machine learning techniques (like GMM) can be useful for regime detection.

---

## What's Next

As I continue learning, I plan to:

- Study portfolio optimization techniques (e.g., Modern Portfolio Theory, Black-Litterman).
- Explore options pricing models like Black-Scholes and binomial trees.
- Try LSTM and reinforcement learning for time-series forecasting and policy-based trading.
- Connect to live data and simulate paper trading using broker APIs (e.g., Alpaca, IBKR).

---

## Final Note

This repository is a snapshot of my personal learning progress in the world of quantitative finance. I'm documenting this as both a study resource and a portfolio of my work. Feedback, contributions, or collaboration ideas are always welcome!
