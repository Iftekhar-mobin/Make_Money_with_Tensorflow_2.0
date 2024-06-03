import time
import logging
import numpy as np
from components.fetch_data import fetch_dataset
from components.preprocessing import process_dataset
from components.model import lstm_model, train_or_reload, predict_prices
from components.constant import n_steps_in, n_steps_out, model_name
from components.visualizaion import plot_predictions

length = 2*(n_steps_in+n_steps_out)

# Configure logging
logging.basicConfig(filename='task.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def visualize_predictions(predicted_price, df_h4):
    collector = []
    for i in range(len(predicted_price)):
    #     print(predicted_price[i, -1:])
        collector.append(predicted_price[i, -1:])
        
    predicted_30 = scaler.inverse_transform(collector[-30:]).reshape(-1)

def task():
    # Your task code goes here
    logging.info("Task executed")

def run_loop():
    while True:
        try:
            dataset = fetch_dataset("EURUSD", length)
            # print(dataset)

            n_features, X , y, _ , _, actual_y = process_dataset(dataset, n_steps_in, n_steps_out, split=False)
            model_arc = lstm_model(n_steps_in, n_steps_out, n_features)
            _, lstm_mod = train_or_reload(model_arc, model_name, X, y)
            prices = predict_prices(lstm_mod, X)
            print(prices)





        except Exception as e:
            logging.error(f"An error occurred: {e}")
        # Sleep for two hours (2 hours * 60 minutes/hour * 60 seconds/minute)
        # time.sleep(2 * 60 * 60)
        time.sleep(3)
        # break

if __name__ == "__main__":
    run_loop()
