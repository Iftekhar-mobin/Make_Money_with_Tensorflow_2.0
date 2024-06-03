import MetaTrader5 as mt5  # pip install MetaTrader5
import pandas as pd  # pip install pandas
from datetime import datetime, timedelta
from components.constant import Login, Password, Server


def fetch_dataset(Symbols, n_bars):
    # Initialize the MT5 connection
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        quit()

    # login to Trade Account with login()
    mt5.login(Login, Password, Server)
    # Choose the symbol and timeframe
    symbol = Symbols
    timeframe = mt5.TIMEFRAME_H1  # Hourly timeframe

    # Define the number of bars you want to retrieve
    num_bars = n_bars

    # Requesting the last hour's data
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)

    # # Shutdown MT5 connection
    # mt5.shutdown()

    # Check if data is retrieved successfully
    if rates is None or len(rates) == 0:
        print("No data, error code =", mt5.last_error())
    else:
        print(f"Retrieved {len(rates)} hourly bars")

    # Convert the rates to a pandas DataFrame
    rates_frame = pd.DataFrame(rates)

    # Convert the time in seconds into a datetime format
    rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')

    # print(rates_frame.head())

    return rates_frame