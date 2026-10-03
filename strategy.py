"""
AAPL equity trading strategy: MACD signals + Bollinger Band midline filter.
60 days of 1-hour data, manual simulation + backtesting.py backtest.

Install:  pip install yfinance pandas numpy matplotlib backtesting
Run:      python strategy.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from backtesting import Backtest, Strategy

# --- 1. Data retrieval and indicators ---------------------------------------
aapl = yf.Ticker("AAPL")
data = aapl.history(period="60d", interval="1h")

# MACD
data["EMA12"] = data["Close"].ewm(span=12, adjust=False).mean()
data["EMA26"] = data["Close"].ewm(span=26, adjust=False).mean()
data["MACD"] = data["EMA12"] - data["EMA26"]
data["Signal_Line"] = data["MACD"].ewm(span=9, adjust=False).mean()

# Bollinger Band midline (20-period SMA), used as risk filter
BB_PERIOD = 20
data["BB_Middle"] = data["Close"].rolling(window=BB_PERIOD).mean()

# Drop initial NaN values caused by indicator windows
data.dropna(inplace=True)

# --- 2. Manual strategy simulation ------------------------------------------
current_position = 0  # 0: none, 1: long, -1: short
entry_price = 0
returns = []
return_indices = []

print("--- Starting Strategy Simulation ---")

for i in range(1, len(data)):
    current = data.iloc[i]
    previous = data.iloc[i - 1]

    # Crossover signals
    macd_buy = (previous["MACD"] < previous["Signal_Line"]) and (current["MACD"] > current["Signal_Line"])
    macd_sell = (previous["MACD"] > previous["Signal_Line"]) and (current["MACD"] < current["Signal_Line"])

    # Risk filters
    bb_buy_filter = current["Close"] < current["BB_Middle"]
    bb_sell_filter = current["Close"] > current["BB_Middle"]

    # Exit long on MACD sell signal
    if current_position == 1 and macd_sell:
        returns.append((current["Close"] / entry_price) - 1)
        return_indices.append(current.name)
        current_position = 0
        entry_price = 0

    # Exit short on MACD buy signal
    elif current_position == -1 and macd_buy:
        returns.append(1 - (current["Close"] / entry_price))
        return_indices.append(current.name)
        current_position = 0
        entry_price = 0

    # Entry if flat (allows immediate reversal)
    if current_position == 0:
        if macd_buy and bb_buy_filter:
            current_position = 1
            entry_price = current["Close"]
        elif macd_sell and bb_sell_filter:
            current_position = -1
            entry_price = current["Close"]

# --- 3. Final calculation and latest signal check ---------------------------
returns_series = pd.Series(returns, index=return_indices)
cumulative_returns = (1 + returns_series).cumprod() - 1

print("\n--- Latest Signal Check ---")
if macd_sell and bb_sell_filter:
    print("Final Bar Signal: SELL (Risk Filter PASSED)")
elif macd_buy and bb_buy_filter:
    print("Final Bar Signal: BUY (Risk Filter PASSED)")
else:
    print("Final Bar Signal: No valid trade")

# --- 4. Equity curve vs buy-and-hold ----------------------------------------
print("\n--- Plotting Return Curve ---")

if not cumulative_returns.empty:
    plt.figure(figsize=(12, 6))

    # Forward-fill the cumulative return across bars with no trade
    equity_curve = pd.Series(np.nan, index=data.index)
    equity_curve.loc[cumulative_returns.index] = cumulative_returns
    equity_curve = equity_curve.ffill().fillna(0.0)

    plt.plot(equity_curve.index, equity_curve * 100, label="Strategy Equity Curve", color="green")

    start_price = data["Close"].iloc[0]
    end_price = data["Close"].iloc[-1]
    buy_hold_return = ((end_price / start_price) - 1) * 100

    plt.axhline(buy_hold_return, color="red", linestyle="--",
                label=f"Buy-and-Hold Return ({buy_hold_return:.2f}%)")
    plt.axhline(0, color="gray", linestyle="-")
    plt.title(f"Equity Curve vs Buy-and-Hold for AAPL ({len(data)} bars)")
    plt.xlabel("Date/Time")
    plt.ylabel("Cumulative Return (%)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.show()
else:
    print("No trades were executed during the simulation period. Cannot plot a return curve.")

# --- 5. Normalised histogram of MACD (PDF estimation) -----------------------
print("\n--- Normalized Histogram (PDF Estimation of MACD) ---")
macd_data = data["MACD"].dropna()

if not macd_data.empty:
    hist, bin_edges = np.histogram(macd_data, bins=30, density=True)
    print(f"Number of data points analyzed: {len(macd_data)}")

    plt.figure(figsize=(10, 6))
    plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor="black", align="edge")
    plt.title("Normalized Histogram of MACD")
    plt.xlabel("MACD Value")
    plt.ylabel("Probability Density")
    plt.grid(axis="y", alpha=0.5)
    plt.show()
else:
    print("MACD data is empty. No histogram will be plotted.")

# --- 6. Sharpe ratio on per-trade returns -----------------------------------
# NOTE: these are per-TRADE returns, not hourly returns, so annualising with
# 252 * 6.5 is only a rough approximation. See the hourly Sharpe in section 8.
print("\n--- Sharpe Ratio Calculation (per-trade returns) ---")
risk_free_rate = 0

if not returns_series.empty:
    annualization_factor = 252 * 6.5
    annualized_average_return = returns_series.mean() * annualization_factor
    annualized_std_dev_return = returns_series.std() * np.sqrt(annualization_factor)

    if annualized_std_dev_return != 0:
        sharpe_ratio = (annualized_average_return - risk_free_rate) / annualized_std_dev_return
        print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    else:
        print("Standard deviation of returns is zero. Cannot calculate Sharpe Ratio.")
else:
    print("No trades were executed. Cannot calculate Sharpe Ratio.")


# --- 7. Backtest with backtesting.py ----------------------------------------
class Algorithm(Strategy):
    ema12_period = 12
    ema26_period = 26
    signal_line_period = 9
    bb_period = 20

    def init(self):
        close = self.data.df["Close"]
        ema12 = close.ewm(span=self.ema12_period, adjust=False).mean()
        ema26 = close.ewm(span=self.ema26_period, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=self.signal_line_period, adjust=False).mean()
        bb_mid = close.rolling(window=self.bb_period).mean()

        self.macd = self.I(lambda x: x, macd)
        self.signal_line = self.I(lambda x: x, signal)
        self.bb_middle = self.I(lambda x: x, bb_mid)

    def next(self):
        current_macd, previous_macd = self.macd[-1], self.macd[-2]
        current_signal, previous_signal = self.signal_line[-1], self.signal_line[-2]
        current_close = self.data.Close[-1]
        current_bb_middle = self.bb_middle[-1]

        macd_buy = (previous_macd < previous_signal) and (current_macd > current_signal)
        macd_sell = (previous_macd > previous_signal) and (current_macd < current_signal)

        bb_buy_filter = current_close < current_bb_middle
        bb_sell_filter = current_close > current_bb_middle

        if self.position.is_long and macd_sell:
            self.position.close()
        elif self.position.is_short and macd_buy:
            self.position.close()
        elif not self.position and macd_buy and bb_buy_filter:
            self.buy()
        elif not self.position and macd_sell and bb_sell_filter:
            self.sell()


bt = Backtest(
    data,
    Algorithm,
    cash=1_500_000,
    commission=0.002,
    exclusive_orders=True,
)
stats = bt.run()
print(stats)

# --- 8. Performance metrics -------------------------------------------------
start, end = stats["Start"], stats["End"]
strat_ret = stats["Return [%]"]
bh_ret = stats["Buy & Hold Return [%]"]

# Sharpe recomputed on hourly equity returns, bars/day taken from the data
equity = stats["_equity_curve"]["Equity"]
r = equity.pct_change().dropna()
bars_per_day = len(data) / data.index.normalize().nunique()
sharpe_hourly = r.mean() / r.std() * np.sqrt(252 * bars_per_day)

print("\n=== AAPL strategy: performance ===")
print(f"Test period:              {start:%Y-%m-%d} -> {end:%Y-%m-%d} ({(end - start).days} days)")
print(f"Total return:             {strat_ret:.2f}%")
print(f"Annualised return:        {stats['Return (Ann.) [%]']:.2f}%")
print(f"Sharpe (backtesting.py):  {stats['Sharpe Ratio']:.2f}")
print(f"Sharpe (hourly bars):     {sharpe_hourly:.2f}")
print(f"Max drawdown:             {stats['Max. Drawdown [%]']:.2f}%")
print(f"Buy & hold:               {bh_ret:.2f}%")
print(f"Difference vs buy & hold: {strat_ret - bh_ret:+.2f} percentage points")

# --- 9. PnL -----------------------------------------------------------------
print("\n--- Profit and Loss (PnL) ---")
initial_cash = 1_500_000
final_equity = stats["Equity Final [$]"]
total_pnl_dollars = final_equity - initial_cash
total_pnl_percentage = (total_pnl_dollars / initial_cash) * 100

print(f"Initial Cash: ${initial_cash:,.2f}")
print(f"Final Equity: ${final_equity:,.2f}")
print(f"Total PnL: ${total_pnl_dollars:,.2f}")
print(f"Total PnL Percentage: {total_pnl_percentage:,.2f}%")

bt.plot()
