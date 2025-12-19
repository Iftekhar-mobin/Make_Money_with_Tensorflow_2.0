# 📈 Make Money with TensorFlow 2.0 – Forex Prediction & Signal Generation

This repository contains a **comprehensive end-to-end research and experimentation framework** for **Forex market prediction, signal generation (BUY/SELL/HOLD), and backtesting** using **classical strategies, machine learning, deep learning, and self-supervised learning (SSL)** techniques built with **TensorFlow 2.x and Python**.

The project is designed for:

* Academic research
* Advanced ML/DL experimentation
* Algorithmic trading prototyping
* Time-series signal labeling and forecasting

---

## 🔑 Key Features

* 📊 **Forex OHLC Data Processing** (H1, H4, multi-currency pairs)
* 📉 **Classical Trading Strategies**

  * Moving Average crossover
  * AutoARIMA forecasting
* 🤖 **Deep Learning Models**

  * LSTM (single & multivariate, multi-step)
  * CNN (Time-Series → Image classification)
* 🧠 **Self-Supervised Learning (SSL)**

  * DeepCluster
  * Contrastive Learning (SimCLR, CPC)
  * LSTM Autoencoders
  * Transformer-based Masked Time-Series Autoencoder
* 🏷️ **Automatic Signal Labeling**

  * BUY / SELL / HOLD
* 🧪 **Backtesting & Portfolio Simulation**
* 📈 **Visualization & Performance Analysis**
* 🔌 **MT5 Integration (Offline & Real-Time)**

---

## 🗂️ Project Structure Overview

```text
.
├── forexPredictor/                 # Core research & experimentation folder
│   ├── notebooks (.ipynb)          # Experiments, training, visualization
│   ├── ohlc_data/                  # Currency-wise OHLC datasets & models
│   ├── signal_labeling/            # SSL & clustering-based labeling methods
│   ├── charts/                     # Generated signal & price visualizations
│   ├── src/                        # Modular pipeline (data, models, utils)
│   ├── main.py                     # Entry script for experiments
│   └── visualization.py            # Plotting utilities
│
├── projects/                       # Structured ML pipeline (final project)
│   ├── datasets/                   # Cleaned & prepared datasets
│   ├── models/                     # Saved ML models & pipelines
│   ├── modules/                    # Feature engineering, validation, simulator
│   └── main.py                     # End-to-end execution pipeline
│
├── matching_excel_internship_result.py
├── README.md
```

---

## 🧠 Methodologies Used

### 1️⃣ Classical Time-Series

* Moving Average strategies
* AutoARIMA forecasting

### 2️⃣ Deep Learning

* LSTM (Single / Multivariate / Multi-step)
* CNN-based Time-Series → Image classification
* Encoder-Decoder architectures

### 3️⃣ Self-Supervised Learning (SSL)

* K-Means + Encoder representations
* LSTM Autoencoder + clustering
* Contrastive Learning (SimCLR / CPC)
* Transformer-based Masked Autoencoding

### 4️⃣ Signal Labeling

* Rule-based labeling
* Cluster-based pseudo-labels
* Hybrid statistical + ML labeling

---

## 🚀 How to Run (Basic)

### 1. Clone the repository

```bash
git clone https://github.com/Iftekhar-mobin/Make_Money_with_Tensorflow_2.0.git
cd Make_Money_with_Tensorflow_2.0
```

### 2. Create environment (recommended)

```bash
conda create -n forex_ml python=3.10
conda activate forex_ml
pip install -r requirements.txt
```

### 3. Run main pipeline

```bash
python forexPredictor/main.py
```

Or explore experiments via **Jupyter Notebooks**:

```bash
jupyter notebook
```

---

## 📊 Data Sources

* Historical Forex OHLC data (H1, H4)
* Currency pairs:

  * EURUSD
  * GBPUSD
  * USDJPY
  * USDCHF
  * USDCAD
  * AUDUSD

⚠️ *Datasets are for research and educational purposes only.*

---

## 📌 Use Cases

* Algorithmic trading research
* Forex signal generation
* Time-series representation learning
* Financial ML & SSL experimentation
* Academic papers & thesis work

---

## ⚠️ Disclaimer

> **This project is strictly for educational and research purposes.**
> It is **NOT financial advice**.
> Trading in financial markets involves risk.

---

## 👤 Author

**Iftekhar Mobin**
Machine Learning & Time-Series Researcher
📧 GitHub: [Iftekhar-mobin](https://github.com/Iftekhar-mobin)

---

## ⭐ Acknowledgements

* TensorFlow & Keras
* scikit-learn
* PyTorch (for SSL concepts)
* MetaTrader 5 (MT5)


