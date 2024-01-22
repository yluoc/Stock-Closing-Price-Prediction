import dash
import dash_core_components as dcc
import dash_html_components as dhc
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objs as pgo
from process_data import new_database
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

new_database.index = new_database.Date
new_database.drop("Date", axis=1, inplace=True)

train_data = new_database[:2000]
valid_data = new_database[2000:]

"""
load trained linear regression model
"""
with open("trained_model/linear_reg_model.pkl", 'rb') as file:
    LR_model = pickle.load(file)

future_data = new_database['Close'].values[2000:].reshape(-1, 1)
LR_pred_price = LR_model.predict(future_data)
valid_data['LR_Predictions'] = LR_pred_price

"""
load trained LSTM model
"""
LSTM_model = load_model("trained_model/LSTM_pred.h5")

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = new_database.values

train_data_lstm = dataset[0:2000, :]
valid_data_lstm = dataset[2000:, :]

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

x_train_data, y_train_data = [], []

for i in range(60, len(train_data_lstm)):
    x_train_data.append(scaled_data[i-60:i, 0])
    y_train_data.append(scaled_data[i, 0])

x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)
x_train_data=np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1))
y_train_data = np.array(y_train_data)

input_data = new_database[len(new_database) - len(valid_data_lstm)-60:].values
input_data = scaler.transform(input_data)

LSTM_model.compile(loss = 'mean_squared_error', optimizer = 'adam')
LSTM_model.fit(x_train_data, y_train_data, epochs=1, batch_size=1, verbose=2)

x_test = []
y_test = new_database['Close'][2000:]
for i in range(60, input_data.shape[0]):
    x_test.append(input_data[i-60:i,0])
x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_closing_price=LSTM_model.predict(x_test)
predicted_closing_price = scaler.inverse_transform(predicted_closing_price)

valid_data['LSTM_Predictions'] = predicted_closing_price

"""
Front-end UI
"""
pred_app = dash.Dash()
server = pred_app.server

pred_app.layout = dhc.Div([
    dhc.H1("stock price analysis dashtable", style = {"textAlign": "center"}),
    dcc.Tabs(id = "tabs", children=[
        dcc.Tab(label='Google stock data', children=[
            dhc.Div([
                dhc.H2("actual closing stock price", style={"textAlign": "center"}),
                dcc.Graph(
                    id = "actual data",
                    figure = {
                        "data":[
                            pgo.Scatter(
                                x = valid_data.index,
                                y = valid_data["Close"],
                                mode = "markers"
                            )
                        ],
                        "layout": pgo.Layout(
                            title = 'scatter plot',
                            xaxis = {'title': 'Date'},
                            yaxis = {'title': 'Closing Rate'}
                        )
                    }
                ),
                dhc.H1("Linear Regression Predicted Closing Price", style = {"textAlign": "center"}),
                dcc.Graph(
                    id = "LR predicted data",
                    figure = {
                        "data": [
                            pgo.Scatter(
                                x = valid_data.index,
                                y = valid_data['LR_Predictions'],
                                mode = 'markers'
                            )
                        ],
                        "layout": pgo.Layout(
                            title = "scatter plot",
                            xaxis = {'title': 'Date'},
                            yaxis = {'title': 'Closing Rate'}
                        )
                    }
                ),
                dhc.H1("LSTM Predicted Closing Price", style={"textAlign": "center"}),
                dcc.Graph(
                    id = "LSTM predicted data",
                    figure = {
                        "data": [
                            pgo.Scatter(
                                x = valid_data.index,
                                y = valid_data['LSTM_Predictions'],
                                mode = 'markers'
                            )
                        ],
                        "layout": pgo.Layout(
                            title = "scatter plot",
                            xaxis = {'title': 'Date'},
                            yaxis = {'title': 'Closing Rate'}
                        )
                    }
                )
            ])
        ])
    ])
])

if __name__ == '__main__':
    pred_app.run_server(debug = True)