import yfinance as yf   # Import yahoo finance 
import math
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.preprocessing.sequence import TimeseriesGenerator

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i + look_back), 0]
		b = dataset[i + look_back, 0]
		dataX.append(a)
		dataY.append(b)
	return np.array(dataX), np.array(dataY)

# load the dataset
dataframe = yf.download('TSLA','2019-01-01','2021-03-08')['Close']
dataset = dataframe.values.reshape(-1, 1)
# print(dataset)

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.95)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# # reshape into X=t and Y=t+1
# look_back = 5
# trainX, trainY = create_dataset(train, look_back)
# testX, testY = create_dataset(test, look_back)

# # reshape input to be [samples, time steps, features]
# # trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
# # testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
# testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

# # create and fit the LSTM network
# model = Sequential()
# model.add(LSTM(4, input_shape=(look_back, 1)))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(trainX, trainY, epochs=50, batch_size=1, verbose=0)

# # make predictions
# trainPredict = model.predict(trainX)
# testPredict = model.predict(testX)

# # # invert predictions
# trainPredict = scaler.inverse_transform(trainPredict)
# trainY = scaler.inverse_transform([trainY])
# testPredict = scaler.inverse_transform(testPredict)
# testY = scaler.inverse_transform([testY])


# # calculate root mean squared error
# trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# print('Train Score: {:.2f} RMSE'.format(trainScore))
# testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# print('Test Score: {:.2f} RMSE'.format(testScore))

# # shift train predictions for plotting
# trainPredictPlot = np.empty_like(dataset)
# trainPredictPlot[:, :] = np.nan
# trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# # shift test predictions for plotting
# testPredictPlot = np.empty_like(dataset)
# testPredictPlot[:, :] = np.nan
# testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict



# # plot baseline and predictions
# train_line, = plt.plot(trainPredictPlot, label='trained')
# test_line, = plt.plot(testPredictPlot, label='predicted')
# pred_line, = plt.plot(scaler.inverse_transform(dataset), label='truth')
# plt.legend(handles=[pred_line, train_line, test_line])
# plt.show()

# for i in range(len(testY[0])):
# 	print(testY[0][i], testPredict[i])


# today = np.array([563])
# allx = np.reshape(today, (today.shape[0], today.shape[1], 1))
# priceTomorrow = model.predict(allx, verbose=0)
# readable_y = scaler.inverse_transform(priceTomorrow)
# print(readable_y)

# # define dataset
# series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# target = np.array([[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11]])
# # # reshape to [10, 1]
# # n_features = 1
# # series = series.reshape((len(series), n_features))
# # # define generator
# n_input = 2
# generator = TimeseriesGenerator(series, target, length=n_input, batch_size=1)

# # define model
# model = Sequential()
# model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
# model.add(Dense(2))
# model.compile(optimizer='adam', loss='mse')
# # fit model
# model.fit_generator(generator, steps_per_epoch=1, epochs=500, verbose=0)
# # make a one step prediction out of sample
# x_input = array([9, 10]).reshape((1, n_input, n_features))
# yhat = model.predict(x_input, verbose=0)
# print(yhat)