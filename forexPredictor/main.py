from matplotlib import pyplot
import numpy as np
import pandas as pd
# To fetch financial data
import yfinance as yf
from sklearn.impute import SimpleImputer
from tqdm import trange
from os import path
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import LSTM, Dense, Dropout

pyplot.style.use('seaborn-darkgrid')


def data_download():
    # Set the ticker as 'EURUSD=X'
    forex_data = yf.download('EURUSD=X', start='2019-01-02', end='2021-12-31')
    # Set the index to a datetime object
    forex_data.index = pd.to_datetime(forex_data.index)
    # Display the last five rows
    forex_data.tail()
    # Set the ticker as 'EURUSD=X'
    forex_data = yf.download('GBPUSD=X', period='120d', interval='1h')
    # Set the index to a datetime object
    forex_data.index = pd.to_datetime(forex_data.index)
    # Display the last five rows
    forex_data.tail()
    forex_data.to_csv('a.csv')


def load_dataset():
    # load dataset
    dataset = pd.read_csv('/home/ifte-home/Downloads/a.csv', header=0, usecols=['Open', 'High', 'Low', 'Close'])

    # dataset.drop('dma', axis=1, inplace=True)
    # print("column number")
    # print(dataset.columns,len(dataset.columns),len(dataset.index))
    # drop columns we don't want to predict
    # deleted-col = list(reframed.columns)[-4 : -1]
    # reframed.drop(deleted-col, axis=1, inplace=True)
    # print("deleted column",deleted-col)
    # print(reframed.columns,len(reframed.columns), len(reframed.index))
    # reframed.drop(reframed.columns[[25,26,27,28,29,5,36,37,38,39,41,42,43,44,45,46,47]], axis=1, inplace=True)
    # print(reframed.head())
    return dataset


def data_cleaning(raw_data):
    dt = raw_data.values
    d = dt.astype(float)

    print("Checkinf for NaN and Inf")
    print("np.nan=", np.where(np.isnan(d)))
    print("is.inf=", np.where(np.isinf(d)))

    print("********************************************")
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(d)
    d = imp.fit_transform(d)

    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(d)
    # print("scaled values", scaled)
    return scaled, scaler


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropna=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropna:
        agg.dropna(inplace=True)

    print("column number")
    print(agg.columns, len(agg.columns), len(agg.index))

    return agg


def train_test_split(reframed, num_hours, col_features):
    # split into train and test sets
    values = reframed.values
    train_size = int(len(values) * 0.8)
    test_size = len(values) - train_size
    train, test = values[0:train_size, :], values[train_size:len(reframed), :]

    # split into input and outputs
    n_obs = num_hours * col_features
    train_X, train_y = train[:, :n_obs], train[:, -col_features]
    test_X, test_y = test[:, :n_obs], test[:, -col_features]
    print(train_X.shape, len(train_X), train_y.shape)

    train_x, test_x = convert_to_3d(train_X, test_X, num_hours, col_features)

    return train_x, train_y, test_x, test_y


def convert_to_3d(train_X, test_X, timestep, num_features):
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], timestep, num_features))
    test_X = test_X.reshape((test_X.shape[0], timestep, num_features))
    return train_X, test_X


def convert_to_2d(test_x_data, hrs, ft):
    return test_x_data.reshape((test_x_data.shape[0], hrs * ft))


def create_model(x_train):
    # design network
    model_seq = Sequential()
    model_seq.add(LSTM(50, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
    model_seq.add(Dropout(0.2))
    model_seq.add(LSTM(20, input_shape=(x_train.shape[1], x_train.shape[2])))
    model_seq.add(Dense(1))
    model_seq.compile(loss='mae', optimizer='adam')
    return model_seq


def operate_model(x_train, y_train, xtest, ytest):
    if path.isfile('/home/ifte-home/Downloads/model.json'):
        model_seq = model_read()
        # model_evaluate(model_seq, x_train, y_train)
        return model_seq

    else:
        bsize = 50
        model_seq = create_model(x_train)

        # fit network
        history = model_seq.fit(x_train, y_train,
                                epochs=20,
                                batch_size=bsize,
                                validation_data=(xtest, ytest),
                                verbose=2,
                                shuffle=False)

        plotting_history(history)
        # model_evaluate(model_seq, x_train, y_train)

        model_write(model_seq)

        return model_seq


def predict(loaded_model, predict_x):
    # make a prediction
    return loaded_model.predict(predict_x)


def model_evaluate(mod, tr_x, train_y):
    # evaluate the model
    scores = mod.evaluate(tr_x, train_y, verbose=0)
    print(mod.metrics_names, scores * 100)


def model_write(model_lstm):
    # serialize model to JSON
    model_json = model_lstm.to_json()
    with open("/home/ifte-home/Downloads/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model_lstm.save_weights("model.h5")
    print("Saved model to disk")


def model_read():
    # load json and create model
    json_file = open('/home/ifte-home/Downloads/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("/home/ifte-home/Downloads/model.h5")
    print("Loaded model from disk")
    return loaded_model


def forecasting_known(test_x, test_y, predicted_y, hours, features, scaler):
    test_x = convert_to_2d(test_x, hours, features)
    # invert scaling for forecast
    inv_predicted_val = np.concatenate((predicted_y, test_x[:, -hours:]), axis=1)
    inv_predicted_val = scaler.inverse_transform(inv_predicted_val)
    inv_predicted_val = inv_predicted_val[:, 0]

    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_x[:, -hours:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    # calculate RMSE
    rems = sqrt(mean_squared_error(inv_y, inv_predicted_val))
    print('Test RMSE: %.3f' % rems)

    return inv_y, inv_predicted_val


def plotting_known_forecast(predicted_y, actual_y):
    pyplot.plot(actual_y)
    pyplot.plot(predicted_y)
    pyplot.show()


def plotting_history(hist):
    # plot history
    pyplot.plot(hist.history['loss'], label='train')
    pyplot.plot(hist.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()


def plot_unknown(predicted_y):
    pyplot.plot(predicted_y)
    pyplot.show()


def forecast_unknown(model, look_back_buffer: np.ndarray, forward_step: int = 1, batch_size: int = 1):
    forecast_predict = np.empty((0, 1), dtype=np.float32)
    forecast_predict[:, :] = np.nan
    print("forecast_predict and lookBbuffer", forecast_predict.shape, look_back_buffer.shape)

    num_elements = look_back_buffer.shape[0]
    time_ahead = look_back_buffer.shape[1]
    feature = look_back_buffer.shape[2]

    flag = 1
    for _ in trange(forward_step, desc='predicting data\t', mininterval=1.0):
        # make prediction with current lookback buffer
        # cur_predict = model.predict(look_back_buffer, batch_size)
        cur_predict = model.predict(look_back_buffer, batch_size)
        print("Current predict", cur_predict.shape)

        if flag:
            flag = 0
            # add prediction to result
            forecast_predict = np.concatenate((forecast_predict, cur_predict))
            print("forecast predict", forecast_predict.shape)

        else:
            # add prediction to result
            forecast_predict = np.concatenate((forecast_predict, cur_predict[-1:]), axis=0)
            print("forecast predict", forecast_predict.shape)

        # ‘C’ means to flatten in row-major (C-style) order.
        look_back_buffer = look_back_buffer.flatten()
        look_back_buffer = np.delete(look_back_buffer, 0, axis=0)
        print("look_back_buffer", look_back_buffer.shape)

        cur_predict = cur_predict.flatten()
        look_back_buffer = np.concatenate((look_back_buffer, cur_predict[-1:]), axis=0)
        look_back_buffer = look_back_buffer.reshape(num_elements, time_ahead, feature)

    return forecast_predict


def runner(num_hours, num_ahead, num_features, time_steps, f_batch_size):
    # step 01
    dataset = load_dataset()
    # step 02
    dataset, scaler = data_cleaning(dataset)
    # step 03
    # frame as supervised learning
    supervised_data = series_to_supervised(dataset, num_hours, num_ahead)
    # step 04
    xtr, ytr, xte, yte = train_test_split(supervised_data, num_hours, num_features)
    # step 06
    model = operate_model(xtr, ytr, xte, yte)
    prediction = predict(model, xte)
    # step 07 forecast known
    input_val, predicted_out = forecasting_known(xte, yte, prediction, num_hours, num_features, scaler)
    # step 08
    plotting_known_forecast(predicted_out, input_val)
    # step 09
    unknown_forecast = forecast_unknown(model, xte, time_steps, f_batch_size)
    # step 10
    plot_unknown(unknown_forecast)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    n_ahead = 1
    n_hours = 3
    n_features = 4
    timestep = 50
    batch_size = 20

    runner(n_hours, n_ahead, n_features, timestep, batch_size)
