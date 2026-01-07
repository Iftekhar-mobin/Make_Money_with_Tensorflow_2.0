from sklearn.cluster import DBSCAN
from ta.trend import MACD, ADXIndicator
from ta.momentum import RSIIndicator, ROCIndicator, StochasticOscillator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator
import numpy as np


def filter_trend_sr_signals(
        df,
        signal_col="Signal",
        sr_th=0.002,      # 0.2% (Forex safe)
        adx_th=20
):
    """
    Filters existing signals:
    BUY only in uptrend + support
    SELL only in downtrend + resistance
    """

    df = df.copy()
    df["FilteredSignal"] = 0

    # ------------------------
    # BUY CONDITIONS
    # ------------------------
    buy_mask = (
        (df[signal_col] == 1) &
        (df["ma_alignment"] == 1) &          # uptrend
        (df["adx"] >= adx_th) &              # strong trend
        (df["dist_to_support"] <= sr_th) &   # near support
        (df["dist_to_resistance"] > sr_th)   # not at resistance
    )

    # ------------------------
    # SELL CONDITIONS
    # ------------------------
    sell_mask = (
        (df[signal_col] == -1) &
        (df["ma_alignment"] == 0) &           # downtrend
        (df["adx"] >= adx_th) &               # strong trend
        (df["dist_to_resistance"] <= sr_th) & # near resistance
        (df["dist_to_support"] > sr_th)       # not at support
    )

    df.loc[buy_mask, signal_col] = 1
    df.loc[sell_mask, signal_col] = -1

    return df


def compute_sr_levels(
        df,
        price_col="close",
        eps_pct=0.002,
        min_samples=20
):
    prices = df[price_col].values.reshape(-1, 1)

    if len(prices) < min_samples:
        return np.array([])

    eps = prices.mean() * eps_pct

    clustering = DBSCAN(
        eps=eps,
        min_samples=min_samples
    ).fit(prices)

    df = df.copy()
    df["sr_cluster"] = clustering.labels_

    clusters = df[df["sr_cluster"] != -1]

    if clusters.empty:
        return np.array([])

    sr_levels = (
        clusters
        .groupby("sr_cluster")[price_col]
        .mean()
        .values
    )

    return sr_levels


def sr_distance_features_safe(
        df,
        sr_levels,
        price_col="close"
):
    prices = df[price_col].values
    max_dist = prices.mean() * 0.05  # 5% cap

    dist_sr = []
    dist_sup = []
    dist_res = []

    for p in prices:
        if len(sr_levels) == 0:
            dist_sr.append(max_dist)
            dist_sup.append(max_dist)
            dist_res.append(max_dist)
            continue

        dists = np.abs(sr_levels - p)
        dist_sr.append(dists.min())

        below = sr_levels[sr_levels <= p]
        above = sr_levels[sr_levels >= p]

        dist_sup.append(
            p - below.max() if len(below) > 0 else max_dist
        )
        dist_res.append(
            above.min() - p if len(above) > 0 else max_dist
        )

    df["dist_to_sr"] = np.array(dist_sr) / prices
    df["dist_to_support"] = np.array(dist_sup) / prices
    df["dist_to_resistance"] = np.array(dist_res) / prices

    return df


def sr_touch_strength_safe(
        df,
        sr_levels,
        tol_pct=0.001
):
    prices = df["close"].values
    tol = prices.mean() * tol_pct

    strength = []

    for p in prices:
        if len(sr_levels) == 0:
            strength.append(0)
        else:
            strength.append(
                np.sum(np.abs(sr_levels - p) < tol)
            )

    df["sr_strength"] = strength
    return df


def extract_sr_cluster_features(
        df,
        window=500,
        step=50
):
    df = df.copy()

    # Initialize features (important)
    for col in [
        "dist_to_sr",
        "dist_to_support",
        "dist_to_resistance",
        "sr_strength"
    ]:
        df[col] = 0.0

    for i in range(window, len(df), step):
        hist = df.iloc[i - window:i]

        sr_levels = compute_sr_levels(hist)

        chunk = df.iloc[i:i + step].copy()

        chunk = sr_distance_features_safe(chunk, sr_levels)
        chunk = sr_touch_strength_safe(chunk, sr_levels)

        df.loc[chunk.index, [
            "dist_to_sr",
            "dist_to_support",
            "dist_to_resistance",
            "sr_strength"
        ]] = chunk[[
            "dist_to_sr",
            "dist_to_support",
            "dist_to_resistance",
            "sr_strength"
        ]]

    # Final safety (never drop rows)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    return df


def extract_fast_features(df):
    df = extract_sr_cluster_features(
        df,
        window=500,  # lookback candles
        step=50  # update frequency
    )

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

    print(df.columns, "\n", len(df))

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
