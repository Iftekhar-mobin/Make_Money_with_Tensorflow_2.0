import numpy as np
from scipy.signal import argrelextrema
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator
from sklearn.linear_model import LinearRegression


def filter_by_slope(df, look_ahead=24, slope_threshold=0):
    print(f"Before: {df['Signal'].value_counts()}")
    lr = LinearRegression()
    signals = df[df['Signal'] != 0].index.tolist()
    for idx in signals:
        pos = df.index.get_loc(idx)
        if pos + look_ahead < len(df):
            y = df['close'].iloc[pos:pos + look_ahead].values.reshape(-1, 1)
            x = np.arange(len(y)).reshape(-1, 1)
            lr.fit(x, y)
            slope = lr.coef_.flatten()[0]
            if df.loc[idx, 'Signal'] == 1 and slope <= slope_threshold:
                df.loc[idx, 'Signal'] = 0
            elif df.loc[idx, 'Signal'] == -1 and slope >= -slope_threshold:
                df.loc[idx, 'Signal'] = 0
    print(f"Filtering out Extra BUY and SELL signal with Linear regression slope values by Zero. "
          f"{df['Signal'].value_counts()}")
    return df
    
# --- Generate extrema signals ---
def generate_signal_only_extrema(df, cluster_length=30):
    df = df.copy()
    
    # Identify local maxima and minima
    max_signal_indices = argrelextrema(df['close'].values, np.greater, order=cluster_length)[0]
    min_signal_indices = argrelextrema(df['close'].values, np.less, order=cluster_length)[0]

    # Initialize Signal column
    df.loc[:, 'Signal'] = 0
    df.loc[max_signal_indices, 'Signal'] = -1  # Sell Signal
    df.loc[min_signal_indices, 'Signal'] = 1   # Buy Signal
    
    # df['Date'] = pd.to_datetime(df['Date'])
    # df.set_index('Date', inplace=True)
    
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
    current_signal = 0
    for idx in df.index:
        if df.loc[idx, 'Signal'] != 0:
            current_signal = df.loc[idx, 'Signal']  # Update the current signal
        df.loc[idx, 'Signal'] = current_signal  # Propagate the current signal
    # Debugging Info
    print(f"Filtered Signals After Propagation: {df['Signal'].value_counts()}")
    return df


def prior_signal_making_zero(df_signal, reset_length=5):
    df = df_signal.copy()
    reset_indexes = set()

    for i in range(1, len(df)):
        if (df.iloc[i]['Signal'] == 1 and df.iloc[i-1]['Signal'] == -1) or \
           (df.iloc[i]['Signal'] == -1 and df.iloc[i-1]['Signal'] == 1):
            for j in range(i, i - reset_length - 1, -1):
                if j >= 0:
                    reset_indexes.add(j)

    df.iloc[list(reset_indexes), df.columns.get_loc('Signal')] = 0
    print(f"After nullify prior {reset_length} signal: {df['Signal'].value_counts()}")
    return df


def shift_signals(df, delay=3):
    """Shift signals forward by delay bars."""
    shifted = df['Signal'].copy()
    df.loc[:, 'Signal'] = 0
    for idx in shifted[shifted != 0].index:
        pos = df.index.get_loc(idx) + delay
        if pos < len(df):
            df.iloc[pos, df.columns.get_loc('Signal')] = shifted.loc[idx]
    print(f"""Shift signals forward by delay {delay} bars.""")
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
