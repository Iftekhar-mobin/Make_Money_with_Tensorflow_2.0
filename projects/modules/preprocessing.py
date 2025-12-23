import pandas as pd
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


def prepare_dataset_for_model(X_selected, y):
    # Refit preprocessing only on selected features
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", MinMaxScaler(feature_range=(0, 1)))
    ])

    X_processed = pipe.fit_transform(X_selected)
    y_mapped = y.map({-1: 1, 0: 0, 1: 2})

    return X_processed, y_mapped, pipe
