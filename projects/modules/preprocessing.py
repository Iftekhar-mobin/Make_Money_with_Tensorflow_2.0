import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer


def rename_col(df):
    # Columns to drop if they exist
    drop_cols = ['spread', 'real_volume']

    # Drop safely
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Convert time → Date if exists
    if 'time' in df.columns:
        df['Date'] = pd.to_datetime(df['time'], unit='s')

    # if 'volume' not in df.columns: df.rename(columns={'open':'Open', 'high':'High', 'low':'Low', 'close':'Close',
    # 'tick_volume':'Volume'}, inplace=True) else: df.rename(columns={'open':'Open', 'high':'High', 'low':'Low',
    # 'close':'Close', 'volume':'Volume'}, inplace=True)

    df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'},
              inplace=True)

    return df


def handling_nan_after_feature_generate(df_f):
    # 1. Drop rows with NA and ensure a deep copy
    df_f_no_NAN = df_f.dropna().copy()

    # 2. Identify non-numeric columns
    non_numeric_cols = df_f_no_NAN.select_dtypes(exclude=['number']).columns

    # 3. Drop non-numeric columns safely
    if len(non_numeric_cols) > 0:
        df_f_no_NAN = df_f_no_NAN.drop(columns=list(non_numeric_cols), errors='ignore')

    return df_f_no_NAN


def prepare_dataset_for_model(X_selected, y, sample_weight=False):
    # Refit preprocessing only on selected features
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", MinMaxScaler(feature_range=(0, 1)))
    ])

    X_processed = pipe.fit_transform(X_selected)
    y_mapped = y.map({-1: 1, 0: 0, 1: 2})

    if not sample_weight:
        return X_processed, y_mapped, pipe
    else:
        return X_processed, y_mapped, pipe, class_weight_balance(y_mapped)


def probability_mapping(proba, decision_threshold=97):
    buy_prob = proba[:, 2]
    sell_prob = proba[:, 1]

    buy_thr = np.percentile(buy_prob, decision_threshold)
    sell_thr = np.percentile(sell_prob, decision_threshold)

    y_pred = np.zeros(len(proba), dtype=int)

    y_pred[(buy_prob >= buy_thr) & (buy_prob > sell_prob)] = 2
    y_pred[(sell_prob >= sell_thr) & (sell_prob > buy_prob)] = 1

    return y_pred


def class_weight_balance(y, clip_max=10.0, normalize=True):
    # Ensure numpy array (safe)
    y = np.asarray(y)

    # Explicit classes
    classes = np.array([0, 1, 2])

    # Compute balanced class weights
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y
    )

    # Class → weight mapping
    class_weight = dict(zip(classes, weights))

    # Build per-sample weights
    sample_weight = np.array([class_weight[int(label)] for label in y])

    # ---- DEBUG PRINTS ----
    print("Class distribution BEFORE:",
          dict(zip(*np.unique(y, return_counts=True))))

    print("Class → Weight mapping:")
    for c in classes:
        print(f"Class {c}: weight = {class_weight[c]}")

    # ---- STABILIZATION (IMPORTANT) ----
    # if clip_max is not None:
    #     sample_weight = np.clip(sample_weight, 1.0, clip_max)

    if normalize:
        sample_weight = sample_weight / sample_weight.mean()

    return sample_weight

