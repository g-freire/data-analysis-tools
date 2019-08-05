"""
LSTM Models for Time Series Forecasting
https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/

This tutorial is divided into four parts; they are:
	Univariate LSTM Models
	Multivariate LSTM Models
	Multi-Step LSTM Models
	Multivariate Multi-Step LSTM Models

This section is divided into six parts; they are:
	Data Preparation
	Vanilla LSTM
	Stacked LSTM
	Bidirectional LSTM
	CNN LSTM
	ConvLSTM

Each of these models are demonstrated for one-step univariate time series forecasting,
but can easily be adapted and used as the input part of a model 
for other types of time series forecasting problems.
"""

# simple univariate(only one variable) lstm example
from numpy import array

# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# split a univariate sequence into samples - using only list slicing traversing
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
# choose a number of time steps
n_steps = 3
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# print summarize the data - 
print("\n-------------------------------------------")
print("Split nivariate sequence into sampels")
print("Input = [10, 20, 30, 40, 50, 60, 70, 80, 90]")
print("Result:")
for i in range(len(X)):
	print(X[i], y[i])
print("-------------------------------------------\n")

"""
Vanilla LSTM
A Vanilla LSTM is an LSTM model that has a single hidden layer of LSTM units,
and an output layer used to make a prediction.
"""
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=200, verbose=0)
# demonstrate prediction
# x_input = array([70, 80, 90])
x_input =array([10, 20, 30, 40, 50, 60, 70, 80, 90])
x_input = x_input.reshape((3, n_steps, n_features))
model_predictions = model.predict(x_input, verbose=0)
print("\n-------------------------------------------")
print("Predictions using LSTM from keras")
print("Input = [10, 20, 30, 40, 50, 60, 70, 80, 90]")
print("Result:")
print(model_predictions)
print("-------------------------------------------\n")
