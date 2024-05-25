import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from pandas import DataFrame 
from sklearn.metrics import mean_absolute_error , mean_squared_error
from numpy import mean , concatenate
from math import sqrt
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
from numpy import array , hstack
import tensorflow as tf
from tensorflow.keras.models import load_model

# from google.colab import drive
# drive.mount('/content/drive')

training_file = 'H4_EURUSD_2015.csv'
df_h1 = pd.read_csv('H1_EURUSD_2015.csv',delimiter=',')
df_h4 = pd.read_csv(training_file,delimiter=',')

# Check for missing values
print(df_h1.isna().sum())

# Drop rows with missing values
df_h1 = df_h1.dropna()

df_h1.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'} ,
             inplace = True)
df_h4.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'} ,
             inplace = True)

df_h1['avg'] = df_h1.loc[:, ['Open', 'High', 'Low', 'Close']].mean(axis=1)

df_h4['avg'] = df_h4.loc[:, ['Open', 'High', 'Low', 'Close']].mean(axis=1)

dataset = df_h4

x_1 = dataset['Close']
x_2 = dataset['Open']
x_3 = dataset['High']
x_4 = dataset['Low']
y = dataset['avg']

x_1 = x_1.values
x_2 = x_2.values
x_3 = x_3.values
x_4 = x_4.values
y = y.values

# convert to [rows, columns] structure
x_1 = x_1.reshape((len(x_1), 1))
x_2 = x_2.reshape((len(x_2), 1))
x_3 = x_2.reshape((len(x_3), 1))
x_4 = x_2.reshape((len(x_4), 1))
y = y.reshape((len(y), 1))

print ("x_1.shape" , x_1.shape)
print ("x_2.shape" , x_2.shape)
print ("y.shape" , y.shape)

# normalization features
scaler = MinMaxScaler(feature_range=(0, 1))
x_1_scaled = scaler.fit_transform(x_1)
x_2_scaled = scaler.fit_transform(x_2)
x_3_scaled = scaler.fit_transform(x_3)
x_4_scaled = scaler.fit_transform(x_4)
y_scaled = scaler.fit_transform(y)

# horizontally stack columns
dataset_stacked = hstack((x_1_scaled, x_2_scaled, x_3_scaled, x_4_scaled, y_scaled))

print ("dataset_stacked.shape" , dataset_stacked.shape)


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out-1
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)


## choose a number of time steps #change this accordingly
n_steps_in, n_steps_out = 60 , 30

# covert into input/output
X, y = split_sequences(dataset_stacked, n_steps_in, n_steps_out)

print ("X.shape" , X.shape)
print ("y.shape" , y.shape)

split = int(len(X)*0.9)
train_X , train_y = X[:split, :] , y[:split, :]
test_X , test_y = X[split:, :] , y[split:, :]

n_features = train_X.shape[2]


print ("train_X.shape" , train_X.shape)
print ("train_y.shape" , train_y.shape)
print ("test_X.shape" , test_X.shape)
print ("test_y.shape" , test_y.shape)
print ("n_features" , n_features)


# define model
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(n_steps_out))
# model.add(Activation('linear'))
model.compile(loss='mse' , optimizer='adam' , metrics=['accuracy'])

model_name = training_file +'__'+ 'TS_model.h5'

try:
    model = load_model(model_name)
except OSError as e:
    print("Model is not loaded, Training model now:", e)
    # Fit network
    history = model.fit(train_X, train_y , epochs=25,  
                        verbose=1,validation_data=(test_X, test_y), 
                        validation_split = 0.1, shuffle=False)
    model.save(model_name)


    # summarize history for accuracy
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
	
    # summarize history for loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
	
predicted_price = model.predict(test_X)