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
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    Xp = pipe.fit_transform(X)
    feature_names = X.columns.tolist()
    return Xp, feature_names, pipe


# --------------------------
# 2. L1 selector
# --------------------------
def l1_selector(X, y, C=0.05):
    model = LogisticRegression(
        penalty="l1", solver="liblinear", C=C, max_iter=3000
    )
    selector = SelectFromModel(model)
    selector.fit(X, y)
    return selector.get_support()


# --------------------------
# 3. Tree-based selector
# --------------------------
def tree_selector(X, y, k=50):
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    selector = SelectKBest(mutual_info_classif, k=min(k, X.shape[1]))
    selector.fit(X, y)
    return selector.get_support()


# --------------------------
# 4. Combined selection
# --------------------------
def combined_feature_selection(X, y, feature_names, k_best):
    l1_mask = l1_selector(X, y)
    tree_mask = tree_selector(X, y, k=k_best)
    mi_mask = SelectKBest(mutual_info_classif, k=k_best).fit(X, y).get_support()

    votes = l1_mask.astype(int) + tree_mask.astype(int) + mi_mask.astype(int)
    selected = np.array(feature_names)[votes >= 2]  # Keep features with ≥2 votes
    masks = {"l1": l1_mask, "tree": tree_mask, "mi": mi_mask}

    return selected, votes, masks


# --------------------------
# 5. Master pipeline
# --------------------------
def select_best_features(X, y, k_best):
    # Preprocess (fixes NaN)
    Xp, feature_names, preproc = preprocess_features(X)

    # Feature selection
    selected, votes, masks = combined_feature_selection(Xp, y, feature_names, k_best)

    return selected, votes, masks, preproc
