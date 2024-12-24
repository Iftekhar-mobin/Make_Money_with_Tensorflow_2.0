import pandas as pd
import numpy as np
from operator import itemgetter
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def feature_generate(data):
    # Initialize the output DataFrame
    indicators_df = data.copy()

    # Moving Averages (Simple and Exponential)
    for window in [5, 10, 20, 50, 100, 200]:
        indicators_df[f'SMA_{window}'] = indicators_df['Close'].rolling(window=window).mean()
        indicators_df[f'EMA_{window}'] = indicators_df['Close'].ewm(span=window, adjust=False).mean()

    # Momentum Indicators
    indicators_df['RSI_14'] = 100 - (100 / (1 + (indicators_df['Close'].diff().clip(lower=0).rolling(window=14).mean() /
                                                 indicators_df['Close'].diff().clip(upper=0).abs().rolling(
                                                     window=14).mean())))

    indicators_df['Stochastic_K'] = ((indicators_df['Close'] - indicators_df['Low'].rolling(14).min()) /
                                     (indicators_df['High'].rolling(14).max() - indicators_df['Low'].rolling(
                                         14).min())) * 100

    indicators_df['Stochastic_D'] = indicators_df['Stochastic_K'].rolling(3).mean()

    # Williams %R
    indicators_df['Williams_%R'] = ((indicators_df['High'].rolling(14).max() - indicators_df['Close']) /
                                    (indicators_df['High'].rolling(14).max() - indicators_df['Low'].rolling(
                                        14).min())) * -100

    # Moving Average Convergence Divergence (MACD)
    indicators_df['MACD'] = indicators_df['Close'].ewm(span=12, adjust=False).mean() - indicators_df['Close'].ewm(
        span=26, adjust=False).mean()
    indicators_df['MACD_Signal'] = indicators_df['MACD'].ewm(span=9, adjust=False).mean()
    indicators_df['MACD_Hist'] = indicators_df['MACD'] - indicators_df['MACD_Signal']

    # Average Directional Index (ADX)
    high_diff = indicators_df['High'].diff()
    low_diff = indicators_df['Low'].diff()
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    true_range = pd.DataFrame({
        'hl': indicators_df['High'] - indicators_df['Low'],
        'hc': (indicators_df['High'] - indicators_df['Close'].shift()).abs(),
        'lc': (indicators_df['Low'] - indicators_df['Close'].shift()).abs()
    }).max(axis=1)
    indicators_df['ATR'] = true_range.rolling(window=14).mean()
    indicators_df['Plus_DI'] = 100 * pd.Series(plus_dm).rolling(window=14).mean() / indicators_df['ATR']
    indicators_df['Minus_DI'] = 100 * pd.Series(minus_dm).rolling(window=14).mean() / indicators_df['ATR']
    indicators_df['ADX'] = 100 * abs(indicators_df['Plus_DI'] - indicators_df['Minus_DI']).rolling(window=14).mean() / (
            indicators_df['Plus_DI'] + indicators_df['Minus_DI'])

    # Commodity Channel Index (CCI)
    typical_price = (indicators_df['High'] + indicators_df['Low'] + indicators_df['Close']) / 3
    indicators_df['CCI'] = (typical_price - typical_price.rolling(20).mean()) / (
            0.015 * typical_price.rolling(20).std())

    # Bollinger Bands
    indicators_df['Bollinger_Mid'] = indicators_df['Close'].rolling(window=20).mean()
    indicators_df['Bollinger_Upper'] = indicators_df['Bollinger_Mid'] + 2 * indicators_df['Close'].rolling(
        window=20).std()
    indicators_df['Bollinger_Lower'] = indicators_df['Bollinger_Mid'] - 2 * indicators_df['Close'].rolling(
        window=20).std()

    # Rate of Change (ROC)
    indicators_df['ROC'] = indicators_df['Close'].pct_change(periods=12) * 100

    # On-Balance Volume (OBV)
    obv = np.where(indicators_df['Close'] > indicators_df['Close'].shift(1), indicators_df['Volume'],
                   np.where(indicators_df['Close'] < indicators_df['Close'].shift(1), -indicators_df['Volume'], 0))
    indicators_df['OBV'] = obv.cumsum()

    # Force Index
    indicators_df['Force_Index'] = indicators_df['Close'].diff(1) * indicators_df['Volume']

    # Accumulation/Distribution Line (ADL)
    adl = ((indicators_df['Close'] - indicators_df['Low']) - (indicators_df['High'] - indicators_df['Close'])) / \
          (indicators_df['High'] - indicators_df['Low']) * indicators_df['Volume']
    indicators_df['ADL'] = adl.cumsum()

    # Money Flow Index (MFI)
    money_flow = typical_price * indicators_df['Volume']
    positive_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
    negative_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)
    mfi_ratio = pd.Series(positive_flow).rolling(14).sum() / pd.Series(negative_flow).rolling(14).sum()
    indicators_df['MFI'] = 100 - (100 / (1 + mfi_ratio))

    # Accumulation Swing Index (ASI)
    indicators_df['ASI'] = ((indicators_df['Close'] - indicators_df['Close'].shift()) +
                            (0.5 * (indicators_df['Close'] - indicators_df['Open'])) +
                            (0.25 * (indicators_df['Close'].shift() - indicators_df['Open'].shift()))) / indicators_df[
                               'ATR']

    # Chaikin Oscillator
    indicators_df['Chaikin_Oscillator'] = indicators_df['ADL'].ewm(span=3, adjust=False).mean() - indicators_df[
        'ADL'].ewm(span=10, adjust=False).mean()

    return indicators_df


def features_selection(df, selection_method='all', topk=10, num_features=225):
    # Separate features and target variable
    X = df.drop(columns=['labels', 'time'])  # Drop non-numeric columns
    y = df['labels']  # Target variable

    # Ensure that only numeric columns are used for correlation
    X_numeric = X.select_dtypes(include=[np.number])

    # Step 1: Correlation Matrix
    # Remove highly correlated features
    cor_matrix = X_numeric.corr().abs()
    upper_triangle = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.85)]
    X.drop(columns=high_corr_features, inplace=True)
    print(f"Features removed due to high correlation: {high_corr_features}")

    # Step 2: Feature Selection with Lasso
    # Scale the data before Lasso
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply LassoCV for feature selection
    lasso = LassoCV(cv=5, random_state=0).fit(X_scaled, y)
    lasso_selected_features = X.columns[lasso.coef_ != 0]
    print(f"Features selected by Lasso: {list(lasso_selected_features)}")

    # Step 3: Recursive Feature Elimination (RFE) with Random Forest
    rfe_selector = RFE(estimator=RandomForestClassifier(), n_features_to_select=10, step=1)
    rfe_selector = rfe_selector.fit(X, y)
    rfe_selected_features = X.columns[rfe_selector.support_]
    print(f"Features selected by RFE: {list(rfe_selected_features)}")

    # Step 4: Feature Importance using Random Forest
    rf = RandomForestClassifier(random_state=0)
    rf.fit(X, y)

    # Combine Lasso and RFE selected features
    selected_features = list(set(lasso_selected_features) | set(rfe_selected_features))
    print(f"Final selected features from Lasso and RFE: {selected_features}")

    list_features = X.columns.tolist()  # List of all feature names

    # Additional feature selection based on ANOVA and Mutual Information
    if selection_method == 'anova' or selection_method == 'all':
        select_k_best_anova = SelectKBest(f_classif, k=topk)
        select_k_best_anova.fit(X, y)
        selected_features_anova = itemgetter(*select_k_best_anova.get_support(indices=True))(list_features)
        print("Selected features by ANOVA:", selected_features_anova)

    if selection_method == 'mutual_info' or selection_method == 'all':
        select_k_best_mic = SelectKBest(mutual_info_classif, k=topk)
        select_k_best_mic.fit(X, y)
        selected_features_mic = itemgetter(*select_k_best_mic.get_support(indices=True))(list_features)
        print("Selected features by Mutual Information:", selected_features_mic)

    # Find common features if selection_method is 'all'
    if selection_method == 'all':
        common = list(set(selected_features_anova).intersection(selected_features_mic))
        print("Common selected features:", len(common), common)

        # Check if enough common features are found
        if len(common) < num_features:
            raise Exception(
                f'Number of common features found ({len(common)}) < {num_features} required features. Increase "topk" variable.')

        # Get indices of common features
        feat_idx = sorted([list_features.index(c) for c in common][:num_features])
        print("Feature indices for common features:", feat_idx)

    return selected_features, feat_idx
