import pandas as pd
from backtesting import Backtest, Strategy


class SignalBandStrategy(Strategy):
    def __init__(self, broker, data, params):
        super().__init__(broker, data, params)
        self.last_signal = None

    def init(self):
        self.last_signal = 0

    def next(self):
        signal = self.data.Signal[-1]

        if signal == 0:
            return  # Hold

        if signal != self.last_signal:
            if self.position:
                self.position.close()

            if signal == 1:  # Buy
                self.buy()
            elif signal == -1:  # Sell
                self.sell()

            self.last_signal = signal


def infer_and_add_date(df, start_date="2000-01-01", candles_per_day=24):
    hours = 24 / candles_per_day
    freq = f"{int(hours)}H"

    df = df.copy()
    df["Date"] = pd.date_range(start=start_date, periods=len(df), freq=freq)
    return df


def run_backtesting_simulator(df, cash=10000, commission=0.002, plot=False):
    if "Date" not in df.columns:
        # df = infer_and_add_date(df)
        raise ValueError("Please provide Date and Time included data for appropriate simulation.")

    if not plot:
        df.set_index(df['Date'], inplace=True)

    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    }

    df.rename(columns=rename_map, inplace=True)

    bt = Backtest(
        df,
        SignalBandStrategy,
        cash=cash,
        commission=commission,
        exclusive_orders=True
    )
    stats = bt.run()

    if plot:
        bt.plot(resample=False)

    return stats
