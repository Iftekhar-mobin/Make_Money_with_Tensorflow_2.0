import numpy as np
from scipy.signal import argrelextrema
from ta.volatility import AverageTrueRange
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
