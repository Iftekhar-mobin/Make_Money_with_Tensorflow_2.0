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


def run_backtesting_simulator(df, cash=10000, commission=0.002, plot=False):
    if "Date" not in df.columns:
        raise ValueError("Please provide Date and Time included data for appropriate simulation.")

    if not plot:
        df.set_index(df['Date'], inplace=True)

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
