import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# Import yahoo finance 
import yfinance as yf   


# Get the data for the stock Apple by specifying the stock ticker, start date, and end date 
dataset_train = yf.download('ACB','2016-01-01','2021-02-04') 
 
# Plot the close prices 
# dataset_train.Close.plot() 
# plt.show()

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

print(dataset_train.head())
look_back=1
train_X, train_Y = create_dataset(dataset_train, look_back)

# print(dataset_train.iloc[:, 1])
# train_X, train_Y = dataset_train.iloc[:, 1:2].values, dataset_train.iloc[:, 1:2].values
# print(training_set)

# sc = MinMaxScaler(feature_range=(0,1))
# training_set_scaled = sc.fit_transform(training_set)




# from tensorflow import keras
# from keras.models import Sequential
# from keras.layers import LSTM
# from keras.layers import Dropout
# from keras.layers import Dense
# model = Sequential()
# model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
# model.add(Dropout(0.2))
# model.add(LSTM(units=50, return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(units=50, return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(units=50))
# model.add(Dropout(0.2))
# model.add(Dense(units=1))
# model.compile(optimizer='adam', loss='mean_squared_error')
# model.fit(X_train, y_train, epochs=100, batch_size=32)

# plt.plot(real_stock_price, color = 'black', label = 'Stock Price')
# plt.plot(predicted_stock_price, color = 'green', label = 'Predicted Stock Price')
# plt.title('Stock Price Prediction')
# plt.xlabel('Time')
# plt.ylabel('Stock Price')
# plt.legend()
# plt.show()