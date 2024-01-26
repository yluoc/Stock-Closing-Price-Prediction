"""
Using LSTM to predict the stock price
"""
from process_data import new_apple_database, new_facebook_database, new_google_database, new_microsoft_database, new_tesla_database
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.models import load_model

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = new_google_database.values

train_data = dataset[0:2000, :]
valid_data = dataset[2000:, :]

new_google_database.index = new_google_database.Date
new_google_database.drop("Date", axis = 1, inplace = True)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset[:, 1:])

x_train_data, y_train_data = [], []

for i in range(60, len(train_data)):
    x_train_data.append(scaled_data[i-60:i, 0])
    y_train_data.append(scaled_data[i, 0])

x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)
x_train_data=np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1))
y_train_data = np.array(y_train_data)

# Build the model

LSTM_model = Sequential()
LSTM_model.add(LSTM(units=50, return_sequences=True, input_shape = (x_train_data.shape[1], 1)))
LSTM_model.add(LSTM(units=50))
LSTM_model.add(Dense(1))

input_data = new_google_database[len(new_google_database) - len(valid_data)-60:].values
input_data = scaler.transform(input_data)

LSTM_model.compile(loss = 'mean_squared_error', optimizer = 'adam')
LSTM_model.fit(x_train_data, y_train_data, epochs=1, batch_size=1, verbose=2)

#Model Test, evaluting the trained model
"""
x_test = []
y_test = new_google_database['Close'][2000:]
for i in range(60, input_data.shape[0]):
    x_test.append(input_data[i-60:i,0])
x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_closing_price=LSTM_model.predict(x_test)
predicted_closing_price = scaler.inverse_transform(predicted_closing_price)

mse = mean_squared_error(y_test, predicted_closing_price)
r2 = r2_score(y_test, predicted_closing_price)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
"""

LSTM_model.save('LSTM_pred.h5')