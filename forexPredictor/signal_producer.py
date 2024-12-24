import numpy as np


def signals_generator(predicted_prices, time_steps, price_diff_new_signal=0.000030):
    # Provided predicted price array
    # predicted_prices = np.array([1.07741435, 1.07797974, 1.07768342, 1.07768771, 1.07819248,
    #                              1.07827098, 1.07808965, 1.07778313, 1.07765595, 1.07710098,
    #                              1.07924897, 1.08013435, 1.0815506, 1.08151388, 1.08143381])

    # Define EMA parameters
    n = time_steps  # Number of periods
    alpha = 2 / (n + 1)

    # Calculate EMA using np.convolve
    ema = np.zeros_like(predicted_prices)
    ema[0] = predicted_prices[0]  # First EMA value is the first price

    for i in range(1, len(predicted_prices)):
        ema[i] = alpha * predicted_prices[i] + (1 - alpha) * ema[i - 1]

    # Initialize signals
    signals = np.zeros(len(predicted_prices))

    # Generate signals based on EMA crossover
    for i in range(1, len(predicted_prices)):
        if predicted_prices[i] > ema[i] and predicted_prices[i - 1] <= ema[i - 1]:
            signals[i] = 1  # Buy signal
        elif predicted_prices[i] < ema[i] and predicted_prices[i - 1] >= ema[i - 1]:
            signals[i] = 2  # Sell signal
        else:
            signals[i] = 0  # Hold signal
        # Print predicted prices at start, buy/sell signals, and end

    start = 0
    end = len(signals) - 1
    important_points = []

    for i in range(len(signals)):
        if i == start or i == end or signals[i] != 0:
            important_points.append((i, predicted_prices[i]))

    pip_change_min = price_diff_new_signal
    # Calculate and print percentage changes
    for j in range(1, len(important_points)):
        index_prev, price_prev = important_points[j - 1]
        index_curr, price_curr = important_points[j]
        #     percent_change = ((price_curr - price_prev) / price_prev) * 100
        pip_change = (price_curr - price_prev)
        if pip_change_min < abs(pip_change):
            # print(f"Change from index {index_prev} to {index_curr}: {pip_change:.4f}")
            pass

        else:
            signals[index_prev] = 0
    #         signals[index_curr] = 0

    i = 0
    while i < len(signals):
        if signals[i] == 1:  # Buy
            while i < len(signals) and (signals[i] == 1 or signals[i] == 0):
                signals[i] = 1
                i += 1
        elif signals[i] == 2:  # Sell
            while i < len(signals) and (signals[i] == 2 or signals[i] == 0):
                signals[i] = 2
                i += 1
        else:
            i += 1
    return signals, np.unique(signals, return_counts=True)