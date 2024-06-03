import pandas as pd
from components.preprocessing import process_dataset
from components.model import lstm_model, train_or_reload, predict_prices
from components.constant import training_file, n_steps_in, n_steps_out, model_name


df = pd.read_csv(training_file,delimiter=',')

n_features, train_X , train_y, test_X , test_y = process_dataset(df, n_steps_in, n_steps_out, split=True)
model_arc = lstm_model(n_steps_in, n_steps_out, n_features)
history, lstm_mod = train_or_reload(model_arc, model_name, train_X, train_y, test_X, test_y)
prices = predict_prices(lstm_mod, test_X)