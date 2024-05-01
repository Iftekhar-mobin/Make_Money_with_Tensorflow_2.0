# LSTM for international airline passengers problem with window regression framing
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tqdm import trange

def make_forecast(model: Sequential, look_back_buffer: numpy.ndarray, timesteps: int=1, batch_size: int=1):
    #forecast_predict = numpy.empty((0, 1,len(look_back_buffer)), dtype=numpy.float32)
    forecast_predict = numpy.empty((0, 1), dtype=numpy.float32)
    #print("forecast_predict and lookBbuffer", forecast_predict,look_back_buffer.shape)
    
    for _ in trange(timesteps, desc='predicting data\t', mininterval=1.0):
        # make prediction with current lookback buffer
        #cur_predict = model.predict(look_back_buffer, batch_size)
        cur_predict = model.predict(look_back_buffer)
        #print("Current predict", cur_predict, cur_predict.shape)

        # add prediction to result
        forecast_predict = numpy.concatenate([forecast_predict, cur_predict], axis=0)
        #print("forecast predict",forecast_predict, forecast_predict.shape)

        # deleted the oldest data from the input array
        look_back_buffer = numpy.delete(look_back_buffer, 0, axis=2)
        #print("look_back_buffer",look_back_buffer,look_back_buffer.shape)
        
        # reshapping the predicted output for concatenate with the input data                   
        cur_predict = numpy.reshape(1,1,1)        
        look_back_buffer = numpy.dstack((look_back_buffer, cur_predict))
        #print("After concat",look_back_buffer)
        
    return forecast_predict

def plot_data(look_back: int,
              train_predict: numpy.ndarray,
              test_predict: numpy.ndarray,
              forecast_predict: numpy.ndarray):
   
    plt.plot([None for _ in range(look_back)] +
             [None for _ in range(look_back)] +
             [None for _ in train_predict] +
             [None for _ in test_predict] +
             [x for x in forecast_predict])
    


    
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
dataframe = read_csv('GBPJPY.csv', header=0, usecols=['Adj Close'])

# drop the nan rows 
dataframe = dataframe.dropna(axis=0, how='any')
#dataframe = read_csv('samples/GBPJPY240.csv', usecols=[1], engine='python', skipfooter=3)

dataset = dataframe.values
dataset = dataset.astype('float32')

mean = numpy.mean(dataset)
sd = numpy.std(dataset)

# removing outliers
final_list = [x for x in dataset if (x > mean - 2 * sd)]
final_list = [x for x in final_list if (x < mean + 2 * sd)]
print(final_list)

dataset = final_list

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.75)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# reshape into X=t and Y=t+1
look_back = 100
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
print(trainX,trainY)
print("Before shaping", trainX.shape,trainY.shape)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
print(trainX,trainY)
print("after reshape", trainX.shape,trainY.shape)

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(50, input_shape=(1, look_back),return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(20))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=25, batch_size=50, verbose=2)
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# generate forecast predictions
print("testing forecast",testX[-1::], testX.shape)
batch_size = 50
forecast_predict = make_forecast(model, testX[-1::], timesteps=20, batch_size=batch_size)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

forecast_predict = scaler.inverse_transform(forecast_predict)

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset), label='Data')
plt.plot(trainPredictPlot, label='trainprediction')
plt.plot(testPredictPlot, label='testprediction')
plot_data(look_back, trainPredict, testPredict, forecast_predict)
#plt.plot(forecast_predict)
plt.show()
