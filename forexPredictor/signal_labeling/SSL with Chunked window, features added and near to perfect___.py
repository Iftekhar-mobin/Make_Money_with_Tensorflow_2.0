import pandas as pd
import random
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.models import load_model
import ta
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")  # suppress ARIMA convergence warnings

# =============================================================================
# # Load data (must contain Open, High, Low, Close, Volume)
# df = pd.read_csv('your_stock_data.csv', parse_dates=['Date'])
# df = df.dropna().reset_index(drop=True)
# 
# =============================================================================


base_dir = r"D:\Repos_git\Make_Money_with_Tensorflow_2.0\forexPredictor"
data_dir = 'ohlc_data'
file_name = '0_61938.csv'

data_path = os.path.join(base_dir, data_dir, file_name)

# Define file and directory names
file_name = '0_61938.csv'
data_dir = 'ohlc_data'
parent_dir = 'forexPredictor'
repo = 'Repos_git'
repo_dir = 'Make_Money_with_Tensorflow_2.0'

# Get base directory (adjust this if needed)
# base_dir = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))  # Go up two levels

# Construct correct full path
# data_path = os.path.join(base_dir, repo, repo_dir, parent_dir, data_dir, file_name)

data_path = os.path.join(base_dir, data_dir, file_name)

# Verify file path
if os.path.exists(data_path):
    print(f"✅ File found: {data_path}")
else:
    print(f"❌ Error: File not found at {data_path}")

# Load CSV
ucols = ['Open', 'High', 'Low', 'Close']
data_main = pd.read_csv(data_path, usecols=ucols)
data_main.reset_index(drop=True, inplace=True)
data_main.head()


window_size = 24
# ================================
# Feature Engineering
# ================================
def compute_slope(series):
    """Computes slope of a time series using linear regression."""
    X = np.arange(len(series)).reshape(-1, 1)
    y = series.values.reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    return model.coef_[0][0]  # slope

def add_technical_indicators(df):
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['EMA_10'] = df['Close'].ewm(span=10).mean()
    df['RSI_14'] = ta.momentum.RSIIndicator(df['Close'], 14).rsi()
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    bb = ta.volatility.BollingerBands(df['Close'])
    df['BB_width'] = bb.bollinger_wband()
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    return df.dropna()

df = add_technical_indicators(data_main)
df.head()
# ================================
# Chunk Time Series
# ================================
def chunk_series(df, window=24):
    chunks = []

    for i in range(len(df) - window):
        chunk = df.iloc[i:i+window]

        # Slope features (trend strength)
        close_slope = compute_slope(chunk['Close'])
        sma_slope = compute_slope(chunk['SMA_10'].fillna(method='bfill'))
        ema_slope = compute_slope(chunk['EMA_10'].fillna(method='bfill'))

        # Feature vector
        features = [
            chunk['log_return'].mean(),                 # 0
            chunk['log_return'].std(),                  # 1
            chunk['SMA_10'].iloc[-1] / chunk['Close'].iloc[-1],  # 2
            chunk['EMA_10'].iloc[-1] / chunk['Close'].iloc[-1],  # 3
            chunk['RSI_14'].iloc[-1],                   # 4
            chunk['ATR'].mean(),                        # 5
            chunk['BB_width'].mean(),                   # 6
            close_slope,                                # 7
            sma_slope,                                  # 8
            ema_slope,                                  # 9
        ]
        chunks.append(features)

    return np.array(chunks)

X = chunk_series(df, window=24)

# Save chunk end timestamps for traceability (optional, for plotting/backtesting)
chunk_timestamps = [df.index[i + window_size - 1] for i in range(len(df) - window_size)]

# ================================
# Normalize Features
# ================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



# 1. Set random seeds
seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# 2. Make TensorFlow deterministic
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = str(seed)


# ================================
# Self-Supervised Autoencoder
# ================================

# Create model save directory
model_dir = os.path.join(base_dir, "saved_models")
os.makedirs(model_dir, exist_ok=True)

# Define model paths
autoencoder_path = os.path.join(model_dir, "autoencoder_model.h5")
encoder_path = os.path.join(model_dir, "encoder_model.h5")

# ============================
# Check if models already exist
# ============================
if os.path.exists(autoencoder_path) and os.path.exists(encoder_path):
    # Load models
    autoencoder = load_model(autoencoder_path)
    encoder = load_model(encoder_path)
    print("✅ Models loaded from disk.")
else:
    # Train new models
    input_dim = X_scaled.shape[1]
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(16, activation='relu')(input_layer)
    encoded = layers.Dense(8, activation='relu')(encoded)
    decoded = layers.Dense(16, activation='relu')(encoded)
    decoded = layers.Dense(input_dim, activation='linear')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    history = autoencoder.fit(X_scaled, X_scaled, epochs=10, batch_size=32, 
                shuffle=False, verbose=1)

    # Create encoder model
    encoder = Model(inputs=input_layer, outputs=encoded)

    # Save models
    autoencoder.save(autoencoder_path)
    encoder.save(encoder_path)
    print(f"✅ Models trained and saved at:\n- {autoencoder_path}\n- {encoder_path}")


    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Autoencoder Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()



# ================================
# Extract Embeddings
# ================================
embeddings = encoder.predict(X_scaled)


# ================================
# Clustering (Buy/Sell/Hold)
# ================================
kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(embeddings)

chunk_sz = step_sz = 24

# Recompute actual future return (5 steps ahead)
future_returns = []
chunk_labels = []

for i in range(0, len(df) - chunk_sz - 5, step_sz):
    end_idx = i + chunk_sz - 1
    ret = (df['Close'].iloc[end_idx + 5] / df['Close'].iloc[end_idx]) - 1
    future_returns.append(ret)
    # chunk_labels.append(clusters[len(future_returns)-1])  # same indexing
    chunk_labels.append(clusters[i])

# cluster_df = pd.DataFrame({'cluster': chunk_labels, 'future_return': future_returns})
cluster_df = pd.DataFrame({
    'timestamp': chunk_timestamps[:len(future_returns)],
    'cluster': chunk_labels,
    'future_return': future_returns
})

# Get average return per cluster
mapping_order = cluster_df.groupby('cluster')['future_return'].mean().sort_values().index.tolist()

# Manual map: lowest = Sell, mid = Hold, highest = Buy
label_map = {mapping_order[0]: 'Buy', mapping_order[1]: 'Hold', mapping_order[2]: 'Sell'}
labels = pd.Series(clusters[:len(future_returns)]).map(label_map)

# =============================================================================
# chart
# =============================================================================
plt.figure(figsize=(15, 6))

# Plot Close price line for the chunk-aligned period
plt.plot(df['Close'].iloc[chunk_sz:chunk_sz+len(labels)].values, label='Close Price', color='black', linewidth=1)

# Plot EMA_10 line (aligned)
plt.plot(df['SMA_10'].iloc[chunk_sz:chunk_sz+len(labels)].values, label='EMA 10', color='orange', linestyle='--')

# Plot scatter signals on top of Close price
plt.scatter(
    range(len(labels)),
    df['Close'].iloc[chunk_sz:chunk_sz+len(labels)].values,
    c=labels.map({'Buy': 'green', 'Sell': 'red', 'Hold': 'gray'}),
    alpha=0.7,
    label='Signals',
    s=50
)

plt.title("Buy/Sell/Hold Classification with Close Price")
plt.xlabel("Time (chunk-aligned index)")
plt.ylabel("Close Price")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# Extend df with the aligned signals
df_signals = df.iloc[chunk_sz:chunk_sz+len(labels)].copy()
df_signals['Signal'] = labels.values

output_csv_path = os.path.join(base_dir, 'signals_output.csv')
df_signals.to_csv(output_csv_path, index=False)
print(f"✅ Signals saved to: {output_csv_path}")

df_loaded = pd.read_csv(output_csv_path)
print("✅ Loaded Data Sample: for visulization")
print(df_loaded.head())



# =============================================================================
# cluster visualization
# =============================================================================
from sklearn.decomposition import PCA
import seaborn as sns


pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], hue=clusters, palette='Set1')
plt.title("KMeans Clusters on Autoencoder Embeddings")
plt.show()


