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
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from numpy import array
from time import sleep, time
from tqdm import trange, tqdm
import tensorflow as tf
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.logging.ERROR)

# simple univariate(only one variable) lstm example
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

print("\n-------------------------------------------")
print("Split univariate(one variable) sequence into sampels")
print("code : seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]")
print("Input = [10, 20, 30, 40, 50, 60, 70, 80, 90]")
print("Result:")
# choose a number of time steps
n_steps = 3
# split into samples
X, y = split_sequence(raw_seq, n_steps)
for i in (range(len(X))):
	#for _ in tqdm(range(1)):
	#	sleep(.1)
		# tqdm.write("Done task %i" % i)
	print(X[i], y[i])
print("-------------------------------------------")

"""
	Vanilla LSTM
	A Vanilla LSTM is an LSTM model that has a single hidden layer of LSTM units,
	and an output layer used to make a prediction.
"""
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
# The model is fit using the efficient Adam version of stochastic gradient descent
# and optimized using the mean squared error, or ‘mse‘ loss function.
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=200, verbose=0)
# demonstrate prediction
# x_input = array([70, 80, 90])
x_input = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
x_input = x_input.reshape((3, n_steps, n_features))
model_predictions = model.predict(x_input, verbose=0)
print("\n-------------------------------------------")
print("Predictions using Vanilla LSTM from keras")
print("model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))")
print("Input = [10, 20, 30, 40, 50, 60, 70, 80, 90]")
print("Result:")
print(model_predictions)
print("-------------------------------------------")

"""
	Stacked LSTM
	Address dimension problem by having the LSTM output a value for each time step in the input data 
	by setting the return_sequences=True argument on the layer. 
	This allows us to have 3D output from hidden LSTM layer as input to the next.
"""

# define model
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True,
               input_shape=(n_steps, n_features)))  # add return sequences
model.add(LSTM(50, activation='relu'))  # add this
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=200, verbose=0)
# demonstrate prediction
# x_input = array([70, 80, 90])
s_x_input = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
stacked_x_input = s_x_input.reshape((3, n_steps, n_features))
stacked_model_predictions = model.predict(stacked_x_input, verbose=0)
print("\n-------------------------------------------")
print("Predictions using Stacked LSTM from keras")
print("model.add(LSTM(50, activation='relu', return_sequences=True,input_shape=(n_steps, n_features)))")
print("Input = [10, 20, 30, 40, 50, 60, 70, 80, 90]")
print("Result:")
print(stacked_model_predictions)
print("-------------------------------------------")


"""
	Bidirectional LSTM
	Address dimension problem by having the LSTM output a value for each time step in the input data 
	by setting the return_sequences=True argument on the layer. 
	This allows us to have 3D output from hidden LSTM layer as input to the next.
"""

from keras.layers import Bidirectional
# define model
model = Sequential()
# uses a bidirectional object before calling LSTM class
model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=200, verbose=0)
# demonstrate prediction
# x_input = array([70, 80, 90])
s_x_input = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
b_x_input = x_input.reshape((3, n_steps, n_features))
bi_model_predictions = model.predict(b_x_input, verbose=0)
print("\n-------------------------------------------")
print("Predictions using Bidirectional LSTM from keras")
print("model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))")
print("Input = [10, 20, 30, 40, 50, 60, 70, 80, 90]")
print("Result:")
print(bi_model_predictions)
print("-------------------------------------------")



