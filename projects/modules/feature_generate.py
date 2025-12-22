import ta
from ta.trend import MACD, ADXIndicator
from ta.momentum import RSIIndicator, ROCIndicator, StochasticOscillator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator
import numpy as np
import pandas as pd


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
# 1️⃣ Extract Original Features
# ================================
def extract_buy_sell_hold_features(df):
    df = df.copy()

    # ========= TREND FEATURES =========
    df["return_5"] = df["close"].pct_change(5)
    df["return_10"] = df["close"].pct_change(10)

    df["sma10"] = df["close"].rolling(10).mean()
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()

    df["ma_alignment"] = (
            (df["sma10"] > df["sma20"]).astype(int) +
            (df["sma20"] > df["sma50"]).astype(int)
    )

    df["slope_20"] = df["close"].rolling(20).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else np.nan
    )

    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    df["adx"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"]).adx()

    # ========= MOMENTUM =========
    df["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi()
    df["roc"] = ta.momentum.ROCIndicator(df["close"], window=10).roc()

    stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"])
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # ========= VOLATILITY =========
    atr = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"])
    df["atr"] = atr.average_true_range()
    df["atr_n"] = df["atr"] / df["close"]

    bb = ta.volatility.BollingerBands(df["close"], window=20)
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / df["close"]

    # ========= MARKET STRUCTURE =========
    df["candle_body"] = abs(df["close"] - df["open"])
    df["candle_range"] = df["high"] - df["low"]
    df["wick_ratio"] = df["candle_range"] / (df["candle_body"] + 1e-9)

    df["hh"] = (df["high"] > df["high"].shift(1)).astype(int)
    df["ll"] = (df["low"] < df["low"].shift(1)).astype(int)

    # ========= SUPPORT & RESISTANCE =========
    df["pivot_high"] = df["high"].rolling(5).apply(lambda x: x[2] == max(x))
    df["pivot_low"] = df["low"].rolling(5).apply(lambda x: x[2] == min(x))

    df["dist_to_high"] = df["close"] - df["high"].rolling(20).max()
    df["dist_to_low"] = df["close"] - df["low"].rolling(20).min()

    df["bb_pos"] = (df["close"] - bb.bollinger_lband()) / (
            (bb.bollinger_hband() - bb.bollinger_lband()) + 1e-9
    )

    # ========= REVERSAL FEATURES =========
    df["doji"] = (df["candle_body"] < df["candle_range"] * 0.1).astype(int)
    df["dir"] = np.sign(df["close"].diff())
    df["dir_change"] = df["dir"].diff().abs()

    # ========= VOLUME FEATURES =========
    df["volume_sma"] = df["volume"].rolling(20).mean()
    df["volume_spike"] = df["volume"] / (df["volume_sma"] + 1e-9)
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()

    # ========= RISK–REWARD FEATURES =========
    df["rr_ratio"] = (df["close"] - df["low"]) / (df["high"] - df["close"] + 1e-9)
    df["signal_strength"] = df["macd_hist"] / (df["atr"] + 1e-9)

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


def add_price_window_features(df, windows):
    df = df.copy()
    price_cols = ["open", "high", "low", "close", "volume"]

    features = {}

    for col in price_cols:
        data = df[col].to_numpy()

        for w in windows:
            roll = pd.Series(data).rolling(w)

            features[f"{col}_mean_{w}"] = roll.mean().to_numpy()
            features[f"{col}_std_{w}"] = roll.std().to_numpy()
            features[f"{col}_min_{w}"] = roll.min().to_numpy()
            features[f"{col}_max_{w}"] = roll.max().to_numpy()

            # 🚀 Fast slope (NO rolling.apply)
            features[f"{col}_slope_{w}"] = rolling_slope_fast(data, w)

    features_df = pd.DataFrame(features, index=df.index)
    return pd.concat([df, features_df], axis=1)


# ================================
# 3️⃣ Full Pipeline
# ================================
def extract_all_features(df, windows=None):
    if windows is None:
        windows = [10, 20, 50, 150]
    print('Please wait working with extract_buy_sell_hold_features ')
    df1 = extract_buy_sell_hold_features(df)
    print('#' * 80)
    # print('Please wait working with add_price_window_features ')
    # df2 = add_price_window_features(df1, windows)
    # print('#' * 80)
    return df1

# ================================
# Usage Example
# ================================
# df = pd.read_csv("ohlc.csv")   # must contain open, high, low, close, volume
# df_features = extract_all_features(df, windows=[3,5,10])
# print(df_features.tail())
