# Trading-stratedy-for-Apple-AAPL-equity
Designed a 60-day Apple Inc. equity strategy integrating technical analysis, volatility modeling, and backtesting to enhance predictive accuracy and risk-adjusted performance.
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Data Retrieval and Indicator Calculation (Required Setup) ---
aapl = yf.Ticker("AAPL")
data = aapl.history(period='60d', interval='1h')

# Calculate MACD
data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = data['EMA12'] - data['EMA26']
data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

# Calculate Bollinger Bands (BB_Middle = 20-period SMA)
BB_PERIOD = 20
data['BB_Middle'] = data['Close'].rolling(window=BB_PERIOD).mean()

# Drop initial NaN values caused by indicator windows
data.dropna(inplace=True)

# --- State Variables for the simulation (Required Setup) ---
current_position = 0  # 0: None, 1: Long, -1: Short
entry_price = 0
returns = []
# Store the index for returns to plot correctly later
return_indices = []

# --- 2. Strategy Simulation (Fixed Logic for Trading) ---
print("--- Starting Strategy Simulation ---")

for i in range(1, len(data)):
    current = data.iloc[i]
    previous = data.iloc[i-1]

    # Crossover Signals
    macd_buy = (previous['MACD'] < previous['Signal_Line']) and (current['MACD'] > current['Signal_Line'])
    macd_sell = (previous['MACD'] > previous['Signal_Line']) and (current['MACD'] < current['Signal_Line'])

    # Risk Filters
    bb_buy_filter = current['Close'] < current['BB_Middle']
    bb_sell_filter = current['Close'] > current['BB_Middle']

    # --- 2a. Exit Logic (Check for current position exit) ---
    trade_exited = False

    # Exit Long position on MACD Sell signal
    if current_position == 1 and macd_sell:
        trade_return = (current['Close'] / entry_price) - 1
        returns.append(trade_return)
        return_indices.append(current.name)
        current_position = 0
        entry_price = 0
        trade_exited = True

    # Exit Short position on MACD Buy signal
    elif current_position == -1 and macd_buy:
        trade_return = 1 - (current['Close'] / entry_price)
        returns.append(trade_return)
        return_indices.append(current.name)
        current_position = 0
        entry_price = 0
        trade_exited = True

    # --- 2b. Entry Logic (Check for new entry if flat, allowing immediate reversal) ---
    if current_position == 0:
        # Long Entry: MACD Buy Signal AND Price is in Risk-Free Zone (below BB Middle)
        if macd_buy and bb_buy_filter:
            current_position = 1
            entry_price = current['Close']

        # Short Entry: MACD Sell Signal AND Price is in Risk-Free Zone (above BB Middle)
        elif macd_sell and bb_sell_filter:
            current_position = -1
            entry_price = current['Close']


# --- 3. Final Calculation and Latest Signal Check ---

# Combine returns with their corresponding timestamp indices
returns_series = pd.Series(returns, index=return_indices)

# Calculate cumulative returns
cumulative_returns = (1 + returns_series).cumprod() - 1

# --- Latest Signal Check (Fixed to use final bar's status) ---
# We use the final iteration's calculated signals/filters
print("\n--- Latest Signal Check ---")
if macd_sell and bb_sell_filter:
    print("Final Bar Signal: SELL (Risk Filter PASSED)")
elif macd_buy and bb_buy_filter:
    print("Final Bar Signal: BUY (Risk Filter PASSED)")
else:
    print("Final Bar Signal: No valid trade")


# --- 4. Plotting the Return Curve (Fixed) ---
print("\n--- Plotting Return Curve ---")

if not cumulative_returns.empty:
    plt.figure(figsize=(12, 6))

    # Align cumulative returns with the historical data index for accurate plotting
    # We must include the starting point (0%) which happens at the first bar of the clean data

    # Create a Series that includes all historical index points, with 0 for non-trade bars
    # and fills forward the last known cumulative return.
    equity_curve = pd.Series(0.0, index=data.index)
    equity_curve.loc[cumulative_returns.index] = cumulative_returns
    equity_curve = equity_curve.replace(to_replace=0.0, method='ffill')

    # Ensure the plot starts at 0% return at the very first bar index
    equity_curve.iloc[0] = 0.0

    plt.plot(equity_curve.index, equity_curve * 100, label='Strategy Equity Curve', color='green')

    # Buy-and-Hold Return Calculation
    start_price = data['Close'].iloc[0]
    end_price = data['Close'].iloc[-1]
    buy_hold_return = ((end_price / start_price) - 1) * 100

    plt.axhline(buy_hold_return, color='red', linestyle='--', label=f'Buy-and-Hold Return ({buy_hold_return:.2f}%)')

    plt.axhline(0, color='gray', linestyle='-')
    plt.title(f'Equity Curve vs Buy-and-Hold for AAPL ({len(data)} bars)')
    plt.xlabel('Date/Time')
    plt.ylabel('Cumulative Return (%)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()
else:
    print("\nNo trades were executed during the simulation period. Cannot plot a return curve.")


# --- 5. Normalized Histogram (PDF Estimation of MACD) (Added to complete the original request) ---
print("\n--- Normalized Histogram (PDF Estimation of MACD) ---")
macd_data = data['MACD'].dropna()

if not macd_data.empty:
    hist, bin_edges = np.histogram(macd_data, bins=30, density=True)
    print(f"Number of data points analyzed: {len(macd_data)}")

    plt.figure(figsize=(10, 6))
    plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor='black')
    plt.title('Normalized Histogram of MACD')
    plt.xlabel('MACD Value')
    plt.ylabel('Probability Density')
    plt.grid(axis='y', alpha=0.5)
    plt.show()
else:
    print("MACD data is empty. No histogram will be plotted.")

# --- 6. Calculate and Display Sharpe Ratio ---
print("\n--- Sharpe Ratio Calculation ---")

# Assuming a risk-free rate of 0 for simplicity in this example (daily data)
# For hourly data, you would need an hourly risk-free rate, which is often approximated as 0.
risk_free_rate = 0

if not returns_series.empty:
    # Annualize the risk-free rate if necessary (not needed if using 0)
    # annual_risk_free_rate = (1 + risk_free_rate)**(252*6.5) - 1 # Example for daily data, adjust for hourly

    # Calculate average daily return
    average_return = returns_series.mean()

    # Calculate standard deviation of daily returns (volatility)
    std_dev_return = returns_series.std()

    # Calculate Sharpe Ratio
    # Annualize the average return and standard deviation for the Sharpe Ratio calculation
    # Assuming 252 trading days * 6.5 hours/day = 1638 trading hours per year
    annualization_factor = 252 * 6.5 # Adjust based on your data's frequency

    annualized_average_return = average_return * annualization_factor
    annualized_std_dev_return = std_dev_return * np.sqrt(annualization_factor)

    if annualized_std_dev_return != 0:
        sharpe_ratio = (annualized_average_return - risk_free_rate) / annualized_std_dev_return
        print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    else:
        print("Standard deviation of returns is zero. Cannot calculate Sharpe Ratio.")
else:
    print("No trades were executed. Cannot calculate Sharpe Ratio.")

%pip install backtesting
from backtesting import Strategy
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

class Algorithm(Strategy):

    # Define parameters for optimization later
    ema12_period = 12
    ema26_period = 26
    signal_line_period = 9
    bb_period = 20
    n1 = 12 # Added for optimization
    n2 = 26 # Added for optimization


    def init(self):
        # Calculate indicators once in the init method
        self.data.df['EMA12'] = self.data.df['Close'].ewm(span=self.ema12_period, adjust=False).mean()
        self.data.df['EMA26'] = self.data.df['Close'].ewm(span=self.ema26_period, adjust=False).mean()
        self.data.df['MACD'] = self.data.df['EMA12'] - self.data.df['EMA26']
        self.data.df['Signal_Line'] = self.data.df['MACD'].ewm(span=self.signal_line_period, adjust=False).mean()
        self.data.df['BB_Middle'] = self.data.df['Close'].rolling(window=self.bb_period).mean()

        # Ensure indicators are accessible as lines
        self.ema12 = self.I(lambda x: x, self.data.df['EMA12'])
        self.ema26 = self.I(lambda x: x, self.data.df['EMA26'])
        self.macd = self.I(lambda x: x, self.data.df['MACD'])
        self.signal_line = self.I(lambda x: x, self.data.df['Signal_Line'])
        self.bb_middle = self.I(lambda x: x, self.data.df['BB_Middle'])


    def next(self):
        # Access latest indicator values
        current_macd = self.macd[-1]
        previous_macd = self.macd[-2]
        current_signal = self.signal_line[-1]
        previous_signal = self.signal_line[-2]
        current_close = self.data.df['Close'][-1]
        current_bb_middle = self.bb_middle[-1]


        # Crossover Signals
        macd_buy = (previous_macd < previous_signal) and (current_macd > current_signal)
        macd_sell = (previous_macd > previous_signal) and (current_macd < current_signal)

        # Risk Filters
        bb_buy_filter = current_close < current_bb_middle
        bb_sell_filter = current_close > current_bb_middle


        # --- Trading Logic ---

        # Exit Long position on MACD Sell signal
        if self.position.is_long and macd_sell:
            self.position.close()

        # Exit Short position on MACD Buy signal
        elif self.position.is_short and macd_buy:
            self.position.close()

        # Enter Long: MACD Buy Signal AND Price is in Risk-Free Zone (below BB Middle)
        elif not self.position and macd_buy and bb_buy_filter:
             self.buy()

        # Enter Short: MACD Sell Signal AND Price is in Risk-Free Zone (above BB Middle)
        elif not self.position and macd_sell and bb_sell_filter:
            self.sell()
      from backtesting import Backtest
bt = Backtest(
    data, # Pass the data as the first argument
    Algorithm, # Pass the class, not an instance
    cash=1500000,
    commission=.002,
    exclusive_orders=True,
)
stats = bt.run()
print(stats)
#PnL calculation
print("\n--- Profit and Loss (PnL) ---")
initial_cash = 1500000 # Use the initial cash value set in the Backtest object
final_equity = stats['Equity Final [$]']
total_pnl_dollars = final_equity - initial_cash
total_pnl_percentage = (total_pnl_dollars / initial_cash)*100

print(f"Initial Cash: ${initial_cash:,.2f}")
print(f"Final Equity: ${final_equity:,.2f}")
print(f"Total PnL: ${total_pnl_dollars:,.2f}")
print(f"Total PnL Percentage: {total_pnl_percentage:,.2f}%")
# Plot the backtest results
bt.plot()
