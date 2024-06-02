import pandas as pd
from components.preprocessing import process_dataset
from components.model import lstm_model, train_or_reload, predict_prices


## choose a number of time steps #change this accordingly
training_file = 'H4_EURUSD_2015.csv'
n_steps_in, n_steps_out = 60 , 30
model_name = training_file +'__'+ 'TS_model.h5'
training_file = 'H4_EURUSD_2015.csv'
# df_h1 = pd.read_csv('H1_EURUSD_2015.csv',delimiter=',')
df = pd.read_csv(training_file,delimiter=',')

n_features, train_X , train_y, test_X , test_y = process_dataset(df, n_steps_in, n_steps_out)
model_arc = lstm_model(n_steps_in, n_steps_out, n_features)
history, lstm_model = train_or_reload(model_arc, model_name, train_X, train_y, test_X, test_y)
prices = predict_prices(lstm_model, test_X)