import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit, KFold, cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report


# ---------------------------------------------------------
# 2) Helper: TimeSeries 5-Fold Evaluation
# ---------------------------------------------------------
def kfold_evaluate(X, y, oversampler, name=""):
    tscv = TimeSeriesSplit(n_splits=5)

    acc_list = []
    f1_list = []

    X_df = pd.DataFrame(X)
    y_df = pd.Series(y).reset_index(drop=True)

    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X_df.iloc[train_idx], X_df.iloc[val_idx]
        y_tr, y_val = y_df.iloc[train_idx], y_df.iloc[val_idx]

        # Oversample inside each training fold
        X_res, y_res = oversampler.fit_resample(X_tr, y_tr)

        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8
        )
        model.fit(X_res, y_res)
        pred = model.predict(X_val)

        acc_list.append(accuracy_score(y_val, pred))
        f1_list.append(f1_score(y_val, pred, average="macro"))

    print(f"\n========== {name} K-FOLD ==========")
    print("Accuracies:", acc_list)
    print("Mean Accuracy:", np.mean(acc_list))
    print("F1 Macro:", f1_list)
    print("Mean F1:", np.mean(f1_list))

    return np.mean(acc_list), np.mean(f1_list)


# ---------------------------------------------------------
# 1) Helper: Train, Evaluate on 20% Test
# ---------------------------------------------------------
def evaluate_oversampler(X_train, y_train, X_test, y_test, oversampler, name=""):
    # Oversample training set only
    X_res, y_res = oversampler.fit_resample(X_train, y_train)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax"
    )
    model.fit(X_res, y_res)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"\n========== {name} (20% Test) ==========")
    print(classification_report(y_test, y_pred))
    print(f"{name} Accuracy: {acc:.4f}")
    print(f"{name} F1 Macro: {f1:.4f}")

    return acc, f1
