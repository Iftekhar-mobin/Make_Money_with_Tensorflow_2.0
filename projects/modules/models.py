import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.model_selection import StratifiedKFold, train_test_split, TimeSeriesSplit, KFold, cross_val_score
from sklearn.metrics import classification_report, f1_score, accuracy_score
from xgboost import XGBClassifier

from modules.validation import evaluate_oversampler, kfold_evaluate


def xgbmodel(X_processed, y_mapped):
    # ------------------------------------
    # 2. 80-20 chronological split
    # ------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_mapped, test_size=0.2, shuffle=False
    )

    # ------------------------------------
    # 3. Model
    # ------------------------------------
    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        eval_metric="mlogloss"
    )

    # ------------------------------------
    # 4. Fit on training data
    # ------------------------------------
    xgb_model.fit(X_train, y_train)

    # ------------------------------------
    # 5. Evaluate on test (20%)
    # ------------------------------------
    y_pred = xgb_model.predict(X_test)
    print("====== 20% Test Classification Report ======")
    print(classification_report(y_test, y_pred))

    return xgb_model


def xgbmodel_adasyn(X_processed, y_mapped):
    # ===============================
    # 2. 80/20 TIME-AWARE SPLIT
    # ===============================
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_mapped, test_size=0.2, shuffle=False
    )

    # ===============================
    # 3. BALANCING MINORITY CLASSES
    # ===============================
    sm = ADASYN()  # SMOTE also works

    X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

    print("Class distribution BEFORE:", y_train.value_counts().to_dict())
    print("Class distribution AFTER :", y_train_bal.value_counts().to_dict())

    # ===============================
    # 4. TRAIN FINAL MODEL
    # ===============================
    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softmax'
    )

    xgb_model.fit(X_train_bal, y_train_bal)

    # ===============================
    # 5. TEST RESULTS (20% DATA)
    # ===============================
    y_pred = xgb_model.predict(X_test)
    print("\n========= 20% TEST SET REPORT after Trained with Data Imbalance handler =========")
    print(classification_report(y_test, y_pred))

    return xgb_model


def xgbmodel_kfold(xgb_model, X_processed, y_mapped):
    # ------------------------------------
    # 6. K-FOLD CROSS-VALIDATION (STRATIFIED)
    #    ➤ 5 folds recommended
    # ------------------------------------

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Accuracy scores
    acc_scores = cross_val_score(
        xgb_model, X_processed, y_mapped, cv=skf, scoring="accuracy"
    )

    # F1 macro scores
    f1_scores = cross_val_score(
        xgb_model, X_processed, y_mapped, cv=skf, scoring="f1_macro"
    )

    print("\n=========== K-FOLD RESULTS ===========")
    print("Accuracy Scores:", acc_scores)
    print("Mean Accuracy:", np.mean(acc_scores))

    print("-------------------------------------")
    print("F1 Macro Scores:", f1_scores)
    print("Mean F1 Macro:", np.mean(f1_scores))
    print("=====================================")


def xgbmodel_comparison_with_adasyn_smote(X_processed, y_mapped):
    # ---------------------------------------------------------
    # 4) Train/Test Split (Time Aware)
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_mapped, test_size=0.2, shuffle=False
    )

    # ---------------------------------------------------------
    # 5) Run ADASYN vs SMOTE on 80/20 + KFold
    # ---------------------------------------------------------
    results = {}

    # 20% Test
    results["ADASYN_20"] = evaluate_oversampler(
        X_train, y_train, X_test, y_test, ADASYN(), "ADASYN"
    )

    results["SMOTE_20"] = evaluate_oversampler(
        X_train, y_train, X_test, y_test, SMOTE(), "SMOTE"
    )

    # TimeSeries K-FOLD
    results["ADASYN_KFOLD"] = kfold_evaluate(X_processed, y_mapped, ADASYN(), "ADASYN")
    results["SMOTE_KFOLD"] = kfold_evaluate(X_processed, y_mapped, SMOTE(), "SMOTE")

    # ---------------------------------------------------------
    # 6) Final Summary & Best Oversampler
    # ---------------------------------------------------------
    print("\n\n================ SUMMARY ================")
    for k, (acc, f1) in results.items():
        print(f"{k} →  Accuracy={acc:.4f},  F1={f1:.4f}")

    best = max(results, key=lambda x: results[x][1])
    print("\n🔥 BEST METHOD (Macro F1):", best)


def predict_with_new_dataset(X_new, pipe, model, test_df_features):
    # pipe = loaded preprocessing pipeline
    X_new_processed = pipe.transform(X_new)
    y_pred = model.predict(X_new_processed)
    unique, counts = np.unique(y_pred, return_counts=True)
    result = dict(zip(unique, counts))
    print(result, len(y_pred), len(test_df_features))

    y_pred = np.array(y_pred).astype(int)
    # Map your 3 classes to (-1, 0, 1)
    mapping = {0: 0, 1: -1, 2: 1}
    y_pred_labels = pd.Series(y_pred).map(mapping)

    # Fix index mismatch
    y_pred_labels.index = test_df_features.index

    # Safe assignment
    test_df_features = test_df_features.copy()
    test_df_features["Signal"] = y_pred_labels

    print(test_df_features.head(), test_df_features["Signal"].value_counts())

    return test_df_features
