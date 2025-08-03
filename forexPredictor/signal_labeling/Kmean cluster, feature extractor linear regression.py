#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
chunk_sz = 24 * 5  # One chunk = 5 days of hourly bars (e.g., 24 hours * 5)
step_sz = 24 * 5   # Step size = non-overlapping
file_name = '0_61938.csv'

# --- Path Setup ---
base_dir = r"D:\Repos_git\Make_Money_with_Tensorflow_2.0\forexPredictor"
data_dir = 'ohlc_data'
data_path = os.path.join(base_dir, data_dir, file_name)

if os.path.exists(data_path):
    print(f"✅ File found: {data_path}")
else:
    print(f"❌ Error: File not found at {data_path}")

# --- Load CSV ---
ucols = ['Open', 'High', 'Low', 'Close']
data_main = pd.read_csv(data_path, usecols=ucols)
data_main.reset_index(drop=True, inplace=True)

# --- Split into chunks ---
def split_time_series(df, chunk_size, step_size):
    chunks = []
    for i in range(0, len(df) - chunk_size + 1, step_size):
        chunk = df.iloc[i:i + chunk_size][['Open', 'High', 'Low', 'Close']].values
        chunks.append(chunk)
    return np.array(chunks)

chunks = split_time_series(data_main, chunk_sz, step_sz)

# --- Trend Feature Extraction ---
def extract_trend_features(window):
    """
    Extracts slope, mean, std for each OHLC column → 12 features per chunk
    """
    features = []
    for i in range(window.shape[1]):
        col = window[:, i]
        x = np.arange(len(col))
        slope = np.polyfit(x, col, 1)[0]
        mean = np.mean(col)
        std = np.std(col)
        features.extend([slope, mean, std])
    return features

trend_features = np.array([extract_trend_features(chunk) for chunk in chunks])

# --- Normalize and Cluster ---
scaler = StandardScaler()
features_scaled = scaler.fit_transform(trend_features)

kmeans = KMeans(n_clusters=3, random_state=0)
cluster_labels = kmeans.fit_predict(features_scaled)

# --- Map Clusters to Buy/Hold/Sell Signals ---
def map_clusters_to_signals(cluster_labels, df, chunk_size, step_size, window=5):
    close_prices = df['Close'].values
    future_returns = []

    for i in range(0, len(df) - chunk_size - window + 1, step_size):
        end_idx = i + chunk_size - 1
        if end_idx + window < len(close_prices):
            ret = (close_prices[end_idx + window] / close_prices[end_idx]) - 1
            future_returns.append(ret)

    future_returns = np.array(future_returns)
    cluster_labels = np.array(cluster_labels[:len(future_returns)])

    grouped_returns = pd.DataFrame({
        'cluster': cluster_labels,
        'return': future_returns
    })
    mean_returns = grouped_returns.groupby('cluster')['return'].mean()

    signal_map = mean_returns.rank().astype(int) - 2  # {-1, 0, 1}
    return signal_map.to_dict()

signal_mapping = map_clusters_to_signals(cluster_labels, data_main, chunk_sz, step_sz)
signals = [signal_mapping[c] for c in cluster_labels]

# --- Plot Entire Series with Signals ---
def plot_full_series_with_signals(df, chunk_size, step_size, signals):
    close_prices = df['Close'].values
    time_index = np.arange(len(close_prices))

    signal_x = []
    signal_y = []
    signal_type = []

    for i, signal in zip(range(0, len(df) - chunk_size + 1, step_size), signals):
        chunk_end_idx = i + chunk_size - 1
        if chunk_end_idx < len(close_prices):
            signal_x.append(chunk_end_idx)
            signal_y.append(close_prices[chunk_end_idx])
            signal_type.append(signal)

    plt.figure(figsize=(16, 6))
    plt.plot(time_index, close_prices, label='Close', color='gray')

    for x, y, sig in zip(signal_x, signal_y, signal_type):
        if sig == 1:
            plt.scatter(x, y, color='blue', label='Buy' if 'Buy' not in plt.gca().get_legend_handles_labels()[1] else "", s=40)
        elif sig == 0:
            plt.scatter(x, y, color='black', label='Hold' if 'Hold' not in plt.gca().get_legend_handles_labels()[1] else "", s=40)
        elif sig == -1:
            plt.scatter(x, y, color='red', label='Sell' if 'Sell' not in plt.gca().get_legend_handles_labels()[1] else "", s=40)

    plt.title('Full Time Series with Buy/Hold/Sell Signals')
    plt.xlabel('Time Index')
    plt.ylabel('Close Price')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_full_series_with_signals(data_main, chunk_size=chunk_sz, step_size=step_sz, signals=signals)

# --- Optional: Visualize Raw Close Price ---
plt.figure(figsize=(20, 8))
data_main['Close'].plot(color='red', linewidth=1)
plt.title('Close Price Over Time', fontsize=18)
plt.xlabel('Time Index')
plt.ylabel('Close Price')
plt.grid(True)
plt.tight_layout()
plt.show()
