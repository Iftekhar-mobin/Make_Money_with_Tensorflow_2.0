from matplotlib.pyplot import plot as plt
import numpy as np


def plotting_signal(prices, signal_values):
    # Plot the results
    plt.figure(figsize=(15, 10))
    plt.plot(prices, linewidth=0.6, color='blue', label='Predicted Prices', )
    # plt.plot(df_h4['avg'].values[-length:], color='red', linewidth=0.3, label='Actaul Prices')
    # plt.plot(ema, label='EMA', linestyle='--', color='green', linewidth=0.3)
    plt.scatter(np.arange(len(prices)), np.where(signal_values == 1, prices, np.nan), marker='^', color='g',
                label='Buy Signal')
    plt.scatter(np.arange(len(prices)), np.where(signal_values == 2, prices, np.nan), marker='v', color='r',
                label='Sell Signal')

    # Annotate the important points
    # for index, price in important_points:
    #     plt.annotate(f'{price:.5f}', (index, price),
    #                  textcoords="offset points", xytext=(0,10), ha='center', color='blue')

    plt.legend()
    plt.title('Predicted Prices and EMA with Buy/Sell Signals and Important Points')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.show()
