import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt

class BackTester():

    def __init__(self, data: pd.DataFrame, capital: float=1000):
        self.capital = capital
        self.price_data = data
        self.analytics_data = data
        self.portfolio_return = None

    def backtest(self) -> float:
        try:
            asset = 0
            profit = []
            market_returns = []
            inPosition = None
            m_asset = self.capital/self.analytics_data['Close'].iloc[0]

            for i in range(len(self.analytics_data)):
                market_returns.append(m_asset*self.analytics_data['Close'].iloc[i] - self.capital)

                if (inPosition is not None):
                    # check if already not in a trade
                    if not inPosition:
                        # check for buy signal if not in trade position
                        if (self.analytics_data['Signal'].iloc[i] == 1):
                            # buy order
                            asset = self.capital / self.analytics_data['Close'].iloc[i]
                            inPosition = True
                            c_profit = 0
                            profit.append(c_profit)

                        # check if it says to wait for a buy signal
                        elif (self.analytics_data['Signal'].iloc[i] == 0):
                            # wait for a buy order, portfolio remains stagnenet profit will remain same as previous
                            profit.append(profit[-1])

                    # while already in a trade
                    else:
                        # check for sell if in trade position
                        if (self.analytics_data['Signal'].iloc[i] == -1):
                            # sell order
                            inPosition = False
                            c_profit = (asset * self.analytics_data['Close'].iloc[i]) - self.capital
                            profit.append(c_profit)

                        # check if it says to hold
                        elif (self.analytics_data['Signal'].iloc[i] == 0):
                            # if already in trade and asks to hold then profit will vary with price
                            c_profit = (asset * self.analytics_data['Close'].iloc[i]) - self.capital
                            profit.append(c_profit)

                else:
                    if (self.analytics_data['Signal'].iloc[i] == 1):
                        # buy order
                        asset = self.capital / self.analytics_data['Close'].iloc[i]
                        inPosition = True
                        profit.append(0)

            for i in range(len(self.analytics_data['Close'])-len(profit)):
                profit.append(profit[-1])

            self.analytics_data['Strategy_return'] = profit
            self.analytics_data['Market_return'] = market_returns
            self.portfolio_return = (sum(profit)/self.capital)*100

            return self.portfolio_return
        except:
            raise Exception

    def backtest_plot(self):
        try:
            # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

            # Plot data on the first subplot
            ax1.plot(self.analytics_data['Date'], self.analytics_data['Market_return'], 'r-', label='Market return')
            #ax1.axhline(0, color='gray', linestyle='--', linewidth=1)  # Add horizontal line at y=0
            ax1.set_xlabel('date')
            ax1.set_ylabel('closes')
            ax1.legend()

            # Plot data on the second subplot
            ax2.plot(self.analytics_data['Date'], self.analytics_data['Strategy_return'], 'b-', label='Strategy return')
            #ax2.axhline(0, color='gray', linestyle='--', linewidth=1)  # Add horizontal line at y=0
            ax2.set_xlabel('date')
            ax2.set_ylabel('profit')
            ax2.legend()

            # Adjust spacing between subplots
            fig.tight_layout()

            # Display the plot
            plt.show()
        except:
            raise Exception

    def add_strategy_signals(self, signal_series: pd.Series):
        # The strategy function must return a list containing signals. 
        # (buy=1, sell=-1 and hold=0) the length must be equal to the number of rows of data.
        self.analytics_data['Signal'] = signal_series

    def calc_max_drawdown(self, data: pd.Series, is_close: bool=False):
        try:
            if not is_close:
                data = data+self.capital
            max_data_stock = data.rolling(window=len(data), min_periods=1).max()
            dd_stock = data/max_data_stock - 1
            max_drawdown = dd_stock.rolling(window=len(data), min_periods=1).min()
            return max_drawdown.min()*100
        except:
            raise Exception


# Run optimization
def rsi_strategy(df, length: int=12, lower: float=30, upper: float=70):
    try:
        # Calculate RSI using pandas-ta
        df['Rsi'] = ta.rsi(df['Close'], length=length)

        # Initialize 'signals' column with 'Hold'
        df['Signal'] = 0

        # Find the first occurrence of 'Buy' and 'Sell' signals
        for i in range(1, len(df)):
            if (df['Rsi'].iloc[i-1] > lower) and (df['Rsi'].iloc[i] <= lower):
                df.at[i, 'Signal'] = 1    # buy
            elif (df['Rsi'].iloc[i-1] < upper) and (df['Rsi'].iloc[i] >= upper):
                df.at[i, 'Signal'] = -1   # sell

        return df['Signal']
    except:
        raise Exception

parameters = {
    'length': (5, 25),
    'lower': (0, 30),
    'upper': (70, 100)}
