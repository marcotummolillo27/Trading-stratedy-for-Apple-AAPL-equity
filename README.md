# AAPL Equity Trading Strategy

A 60-day algorithmic trading strategy for Apple Inc. (AAPL) built in Python, combining **MACD-based signal generation**, **Bollinger Band risk filtering**, and **full backtesting** via the `backtesting.py` framework on 1-hour intraday data.

---

## Overview

This project implements a systematic long/short trading strategy on AAPL. The pipeline covers data retrieval, indicator computation, signal generation with risk filtering, backtested performance evaluation, and Sharpe ratio calculation.

**Core logic:**
- Enter **long** when MACD crosses above its signal line *and* price is below the 20-period Bollinger Band midline
- Enter **short** when MACD crosses below its signal line *and* price is above the 20-period Bollinger Band midline
- Exit on the opposite MACD crossover

---

## Strategy Architecture

### 1. Data Retrieval

```python
import yfinance as yf

aapl = yf.Ticker("AAPL")
data = aapl.history(period='60d', interval='1h')
```

60 days of hourly AAPL price data (~600 bars) pulled via `yfinance`.

---

### 2. Indicators

#### MACD

```python
data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD']  = data['EMA12'] - data['EMA26']
data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
```

| Component | Description |
|---|---|
| EMA12 | 12-period EMA — fast line |
| EMA26 | 26-period EMA — slow line |
| MACD | EMA12 − EMA26 |
| Signal Line | 9-period EMA of MACD |

#### Bollinger Band midline (risk filter)

```python
BB_PERIOD = 20
data['BB_Middle'] = data['Close'].rolling(window=BB_PERIOD).mean()
```

Used as a trend filter: longs are only taken below the midline, shorts only above it.

---

### 3. Signal Logic

```python
# Crossover detection
macd_buy  = (previous['MACD'] < previous['Signal_Line']) and (current['MACD'] > current['Signal_Line'])
macd_sell = (previous['MACD'] > previous['Signal_Line']) and (current['MACD'] < current['Signal_Line'])

# Risk filters
bb_buy_filter  = current['Close'] < current['BB_Middle']
bb_sell_filter = current['Close'] > current['BB_Middle']
```

| Signal | Condition |
|---|---|
| Long entry | MACD bullish crossover AND price < BB midline |
| Short entry | MACD bearish crossover AND price > BB midline |
| Long exit | MACD bearish crossover |
| Short exit | MACD bullish crossover |

---

### 4. Backtesting

Implemented using the `backtesting.py` framework with an `Algorithm` class inheriting from `Strategy`:

```python
from backtesting import Backtest, Strategy

bt = Backtest(
    data,
    Algorithm,
    cash=1_500_000,
    commission=0.002,
    exclusive_orders=True,
)
stats = bt.run()
```

**P&L calculation:**

```python
initial_cash  = 1_500_000
final_equity  = stats['Equity Final [$]']
total_pnl     = final_equity - initial_cash
total_pnl_pct = (total_pnl / initial_cash) * 100
```

---

### 5. Performance Metrics

#### Sharpe Ratio

```python
annualization_factor        = 252 * 6.5   # 1638 trading hours/year
annualized_return           = returns.mean() * annualization_factor
annualized_vol              = returns.std()  * np.sqrt(annualization_factor)
sharpe_ratio                = annualized_return / annualized_vol
```

#### Outputs

- Equity curve vs buy-and-hold benchmark (matplotlib)
- Normalised MACD histogram (PDF estimation)
- `backtesting.py` interactive performance plot

---

## Tech Stack

| Library | Role |
|---|---|
| `yfinance` | Market data retrieval |
| `pandas` | Data manipulation and indicator calculation |
| `numpy` | Numerical operations and annualisation |
| `matplotlib` | Equity curve and histogram plotting |
| `backtesting.py` | Strategy backtesting framework |

---

## Installation

```bash
git clone https://github.com/marcotummolillo27/Trading-strategy-for-Apple-AAPL-equity.git
cd Trading-strategy-for-Apple-AAPL-equity
pip install yfinance pandas numpy matplotlib backtesting
python strategy.py
```

---

## Project Structure

```
├── strategy.py    # Data retrieval, indicators, simulation, Sharpe ratio, plots
└── README.md
```

---

## About

Built as part of an independent exploration of quantitative finance and algorithmic trading.  
**Marco Tummolillo** — Actuarial Science, University of Amsterdam  
[GitHub](https://github.com/marcotummolillo27)

> *This project is for educational purposes only and does not constitute financial advice.*
