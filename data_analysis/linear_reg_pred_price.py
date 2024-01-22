"""
Build linear regression model
Using the first 2000 line fo close price to train the model,
using the trained model to predit the rest line of close price
"""
from process_data import new_database
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

x = new_database['Close'].values.reshape(-1, 1)
y = new_database['Close']

x_train = x[:2000]
x_test = x[2000:]
y_train = y[:2000]
y_test = y[2000:]

linear_regression_model = LinearRegression()
linear_regression_model.fit(x_train, y_train)

future_x = new_database['Close'].values[2000:].reshape(-1, 1) 
future_y_pred = linear_regression_model.predict(future_x)

# save trained linear regression model into pickle file

linear_reg_model = "trained_model/linear_reg_model.pkl"

with open(linear_reg_model, 'wb') as file:
    pickle.dump(linear_regression_model, file)

#r2 = r2_score(y_test, future_y_pred)
#print(f"r2: {r2}")