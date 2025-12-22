from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import numpy as np


# --------------------------
# 1. Preprocessing
# --------------------------
def preprocess_features(X):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),  # faster & robust
        ("scaler", StandardScaler(with_mean=False))  # faster for large data
    ])
    Xp = pipe.fit_transform(X)
    return Xp, X.columns.to_numpy(), pipe


# --------------------------
# 2. L1 selector (FAST)
# --------------------------
def l1_selector(X, y, C=0.1):
    model = LogisticRegression(
        penalty="l1",
        solver="saga",
        C=0.1,
        max_iter=2000,  # ⬆ small increase
        tol=1e-3,  # ⬅ looser tolerance (faster)
        n_jobs=-1
    )

    model.fit(X, y)
    return np.abs(model.coef_).sum(axis=0) > 1e-6


# --------------------------
# 3. Mutual Information (ONCE)
# --------------------------
def mi_selector(X, y, k):
    mi = mutual_info_classif(X, y, n_neighbors=3, random_state=42)
    idx = np.argsort(mi)[-k:]
    mask = np.zeros(X.shape[1], dtype=bool)
    mask[idx] = True
    return mask


# --------------------------
# 4. Random Forest selector (FAST)
# --------------------------
def rf_selector(X, y, k):
    rf = RandomForestClassifier(
        n_estimators=80,  # 🔥 reduced
        max_depth=12,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X, y)
    idx = np.argsort(rf.feature_importances_)[-k:]
    mask = np.zeros(X.shape[1], dtype=bool)
    mask[idx] = True
    return mask


# --------------------------
# 5. Combined selection
# --------------------------
def combined_feature_selection(X, y, feature_names, k):
    l1_mask = l1_selector(X, y)
    mi_mask = mi_selector(X, y, k)
    rf_mask = rf_selector(X, y, k)

    votes = l1_mask.astype(int) + mi_mask.astype(int) + rf_mask.astype(int)

    selected = feature_names[votes >= 2]
    masks = {"l1": l1_mask, "mi": mi_mask, "rf": rf_mask}

    return selected, votes, masks


# --------------------------
# 6. Master pipeline
# --------------------------
def select_best_features(X, y, k_best):
    Xp, feature_names, preproc = preprocess_features(X)
    selected, votes, masks = combined_feature_selection(Xp, y, feature_names, k_best)
    return selected, votes, masks, preproc
