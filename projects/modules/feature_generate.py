from ta.trend import MACD, ADXIndicator
from ta.momentum import RSIIndicator, ROCIndicator, StochasticOscillator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator
import numpy as np


def extract_fast_features(df):
    df = df.copy()

    df["return_10"] = df["close"].pct_change(10)

    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()
    df["ma_alignment"] = (df["sma20"] > df["sma50"]).astype(int)

    df["slope_20"] = df["close"].rolling(20).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
    )

    macd = MACD(df["close"])
    df["macd_hist"] = macd.macd_diff()

    df["adx"] = ADXIndicator(df["high"], df["low"], df["close"]).adx()

    df["rsi"] = RSIIndicator(df["close"]).rsi()
    df["roc"] = ROCIndicator(df["close"], window=10).roc()

    stoch = StochasticOscillator(df["high"], df["low"], df["close"])
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    atr = AverageTrueRange(df["high"], df["low"], df["close"])
    df["atr_n"] = atr.average_true_range() / df["close"]

    bb = BollingerBands(df["close"])
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / df["close"]
    df["bb_pos"] = (df["close"] - bb.bollinger_lband()) / (
            bb.bollinger_hband() - bb.bollinger_lband() + 1e-9
    )

    df["obv"] = OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()

    return df


# ================================
# 2️⃣ Add Sliding-Window Features for ALL Numeric Columns
# ================================

def rolling_slope_fast(arr, window):
    """
    Fast rolling linear regression slope using analytical formula.
    """
    n = window
    x = np.arange(n)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    slopes = np.full(len(arr), np.nan)

    for i in range(n - 1, len(arr)):
        y = arr[i - n + 1:i + 1]
        y_mean = y.mean()
        slopes[i] = ((x - x_mean) * (y - y_mean)).sum() / x_var

    return slopes

