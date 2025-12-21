import numpy as np
from scipy.signal import argrelextrema
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator
from modules.chart import generate_signal_plot


def prepare_signal(raw_data):
    dataset = filter_by_slope(
        remove_low_volatility_signals(
            prior_signal_making_zero(
                signal_propagate(
                    shift_signals(raw_data)
                )
            )
        )
    )
    return dataset


def visualize_dataset(df, processed, limit=3000):
    df.reset_index(inplace=True, drop=True)
    generate_signal_plot(df, val_limit=limit)
    generate_signal_plot(generate_signal_only_extrema(df), val_limit=limit)
    generate_signal_plot(shift_signals(df), val_limit=limit)
    generate_signal_plot(signal_propagate(shift_signals(df)), val_limit=limit)

    # processed = remove_low_volatility_signals(
    #     prior_signal_making_zero(
    #         signal_propagate(
    #             shift_signals(df)
    #         )
    #     )
    # )
    generate_signal_plot(processed, val_limit=limit)
    generate_signal_plot(filter_by_slope(processed), val_limit=limit)
    generate_signal_plot(filter_by_slope(processed, look_ahead=30), val_limit=limit)


def filter_by_slope(df, look_ahead=24, slope_threshold=0):
    df = df.copy()
    print(f"Before: {df['Signal'].value_counts()}")

    s = df["Signal"].to_numpy()
    close = df["close"].to_numpy()

    n = look_ahead
    x = np.arange(n)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    signal_idx = np.where(s != 0)[0]

    for i in signal_idx:
        if i + n >= len(close):
            continue

        y = close[i:i+n]
        y_mean = y.mean()

        # Fast LR slope
        slope = ((x - x_mean) * (y - y_mean)).sum() / x_var

        if s[i] == 1 and slope <= slope_threshold:
            s[i] = 0
        elif s[i] == -1 and slope >= -slope_threshold:
            s[i] = 0

    df["Signal"] = s
    print(
        "Filtering BUY/SELL by slope: ",
        np.unique(s, return_counts=True)
    )
    return df


def generate_signal_only_extrema(df, cluster_length=30):
    df = df.copy()

    close = df["close"].to_numpy()
    signal = np.zeros(len(df), dtype=np.int8)

    max_idx = argrelextrema(close, np.greater, order=cluster_length)[0]
    min_idx = argrelextrema(close, np.less, order=cluster_length)[0]

    signal[max_idx] = -1   # Sell
    signal[min_idx] = 1    # Buy

    df["Signal"] = signal
    return df


def generate_consecutive_signal_label(df, col='Signal'):
    df = df.copy()
    signal = df[col].to_numpy(dtype=int)

    # --- Step 1: Propagate last non-zero signal forward ---
    propagated_signal = np.zeros_like(signal)
    last_value = 0
    for i, current_value in enumerate(signal):
        if current_value != 0:
            last_value = current_value
        propagated_signal[i] = last_value

    return propagated_signal


# Propagate signals consecutively
def signal_propagate(df_signals):
    df = df_signals.copy()

    s = df["Signal"]
    df["Signal"] = s.replace(0, np.nan).ffill().fillna(0).astype(int)

    print(f"Filtered Signals After Propagation: {df['Signal'].value_counts()}")
    return df


def prior_signal_making_zero(df_signal, reset_length=5):
    df = df_signal.copy()

    s = df["Signal"].to_numpy()
    s_new = s.copy()

    # Detect sign changes (1 → -1 or -1 → 1)
    flip_idx = np.where(s[1:] * s[:-1] == -1)[0] + 1

    for i in flip_idx:
        start = max(0, i - reset_length)
        s_new[start:i+1] = 0

    df["Signal"] = s_new
    print(f"After nullify prior {reset_length} signal: {np.unique(s_new, return_counts=True)}")
    return df


def shift_signals(df, delay=3):
    """
    Shift non-zero signals forward by `delay` bars (fast, vectorized).
    """
    df = df.copy()

    s = df["Signal"].to_numpy()
    shifted = s.copy()

    # Reset all signals
    s[:] = 0

    if delay < len(s):
        s[delay:] = shifted[:-delay]

    df["Signal"] = s
    print(f"Shift signals forward by delay {delay} bars.")
    return df


def remove_low_volatility_signals(df, threshold_percentile=20, atr_period=14):
    """
    Set signals to 0 when ATR is below a certain percentile threshold (low volatility).
    """
    df = df.copy()

    # Check required columns
    for col in ['high', 'low', 'close', 'Signal']:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Calculate ATR
    atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=atr_period)
    df['ATR'] = atr.average_true_range()

    # Calculate volatility threshold
    atr_threshold = df['ATR'].quantile(threshold_percentile / 100)

    # Mask: where ATR is too low, nullify the signal
    low_volatility = df['ATR'] < atr_threshold
    df.loc[low_volatility, 'Signal'] = 0

    print(f"Low-volatility threshold (ATR percentile {threshold_percentile}%) = {atr_threshold:.6f}")
    print(f"After nullify prior signal: {df['Signal'].value_counts()}")
    return df


def generate_atr_sma_signals(df, atr_period=14, atr_multiplier=1.5, sma_period=50, low_vol_percentile=20):
    """
    ATR-based breakout signals with SMA trend filter and low-volatility filter.

    df: DataFrame with 'high', 'low', 'close' columns
    atr_period: ATR lookback period
    atr_multiplier: multiplier for ATR breakout
    sma_period: SMA period for trend filter
    low_vol_percentile: percentile to remove low-volatility signals
    """
    df = df.copy()

    # ---------------------
    # ATR Calculation
    # ---------------------
    atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=atr_period)
    df['ATR'] = atr.average_true_range()

    # ---------------------
    # SMA Trend Filter
    # ---------------------
    df['SMA'] = df['close'].rolling(sma_period).mean()

    # ---------------------
    # ATR Breakout Bands
    # ---------------------
    df['Upper_Band'] = df['close'].shift(1) + atr_multiplier * df['ATR']
    df['Lower_Band'] = df['close'].shift(1) - atr_multiplier * df['ATR']

    # ---------------------
    # Initial Signals
    # ---------------------
    df['Signal'] = 0
    df.loc[df['close'] > df['Upper_Band'], 'Signal'] = 1   # Buy
    df.loc[df['close'] < df['Lower_Band'], 'Signal'] = -1  # Sell

    # ---------------------
    # Trend Filter
    # ---------------------
    df.loc[(df['Signal'] == 1) & (df['close'] < df['SMA']), 'Signal'] = 0
    df.loc[(df['Signal'] == -1) & (df['close'] > df['SMA']), 'Signal'] = 0

    # ---------------------
    # Low-volatility Filter
    # ---------------------
    atr_threshold = df['ATR'].quantile(low_vol_percentile / 100)
    df.loc[df['ATR'] < atr_threshold, 'Signal'] = 0

    print(f"Low-volatility threshold (ATR percentile {low_vol_percentile}%) = {atr_threshold:.6f}")
    print(f"Signal counts after trend & volatility filter:\n{df['Signal'].value_counts()}")

    # ---------------------
    # Final Output
    # ---------------------
    return df[['close', 'high', 'low', 'ATR', 'SMA', 'Upper_Band', 'Lower_Band', 'Signal']]


def ema_crossover_signal(df, fast=9, slow=21):
    df = df.copy()
    df['EMA_fast'] = df['close'].ewm(span=fast, adjust=False).mean()
    df['EMA_slow'] = df['close'].ewm(span=slow, adjust=False).mean()
    
    df['Signal'] = 0
    df.loc[df['EMA_fast'] > df['EMA_slow'], 'Signal'] = 1
    df.loc[df['EMA_fast'] < df['EMA_slow'], 'Signal'] = -1
    
    # Optional: remove conflicting signals with trend SMA(50)
    df['SMA50'] = df['close'].rolling(50).mean()
    df.loc[(df['Signal']==1) & (df['close']<df['SMA50']), 'Signal']=0
    df.loc[(df['Signal']==-1) & (df['close']>df['SMA50']), 'Signal']=0
    
    return df[['close','EMA_fast','EMA_slow','SMA50','Signal']]


def bollinger_signal(df, period=20, std_mult=2):
    df = df.copy()
    df['SMA'] = df['close'].rolling(period).mean()
    df['STD'] = df['close'].rolling(period).std()
    df['Upper'] = df['SMA'] + std_mult*df['STD']
    df['Lower'] = df['SMA'] - std_mult*df['STD']
    
    df['Signal'] = 0
    df.loc[df['close'] < df['Lower'], 'Signal'] = 1   # Buy
    df.loc[df['close'] > df['Upper'], 'Signal'] = -1  # Sell
    
    return df[['close','SMA','Upper','Lower','Signal']]


def rsi_signal(df, period=14, lower=30, upper=70):
    df = df.copy()
    df['RSI'] = RSIIndicator(df['close'], period).rsi()
    
    df['Signal'] = 0
    df.loc[df['RSI'] < lower, 'Signal'] = 1
    df.loc[df['RSI'] > upper, 'Signal'] = -1
    
    # Trend filter optional
    df['SMA50'] = df['close'].rolling(50).mean()
    df.loc[(df['Signal']==1) & (df['close']<df['SMA50']), 'Signal']=0
    df.loc[(df['Signal']==-1) & (df['close']>df['SMA50']), 'Signal']=0
    
    return df[['close','RSI','SMA50','Signal']]


def large_engulfing_signal(df, atr_period=14, sma_period=50, min_body_multiplier=0.5):
    """
    Detect large bullish/bearish engulfing candles and generate signals
    df: DataFrame with 'open', 'high', 'low', 'close'
    atr_period: ATR lookback for filtering small candles
    sma_period: SMA period for trend filter
    min_body_multiplier: min candle body size relative to ATR
    """
    df = df.copy()

    # ---------------------
    # ATR filter for candle size
    # ---------------------
    atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=atr_period)
    df['ATR'] = atr.average_true_range()
    
    # Compute candle body
    df['Body'] = abs(df['close'] - df['open'])

    # SMA trend filter
    df['SMA'] = df['close'].rolling(sma_period).mean()

    # Initialize signal
    df['Signal'] = 0

    # Loop over candles to detect engulfing
    for i in range(1, len(df)):
        prev_open = df.loc[df.index[i-1], 'open']
        prev_close = df.loc[df.index[i-1], 'close']
        curr_open = df.loc[df.index[i], 'open']
        curr_close = df.loc[df.index[i], 'close']
        curr_body = df.loc[df.index[i], 'Body']
        curr_atr = df.loc[df.index[i], 'ATR']

        # Minimum body filter
        if curr_body < curr_atr * min_body_multiplier:
            continue

        # Bullish Engulfing
        if (curr_close > curr_open) and (curr_close > prev_open) and (curr_open < prev_close):
            # Trend filter
            if curr_close > df.loc[df.index[i], 'SMA']:
                df.loc[df.index[i], 'Signal'] = 1

        # Bearish Engulfing
        elif (curr_close < curr_open) and (curr_close < prev_open) and (curr_open > prev_close):
            if curr_close < df.loc[df.index[i], 'SMA']:
                df.loc[df.index[i], 'Signal'] = -1

    print(f"Engulfing signals generated: {df['Signal'].value_counts()}")
    return df[['open','high','low','close','ATR','SMA','Body','Signal']]
