import dash
from dash import dcc
from dash import html
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objs as pgo
from process_data import new_google_database
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

new_google_database.index = new_google_database.Date
new_google_database.drop("Date", axis=1, inplace=True)

train_data = new_google_database[:2000]
valid_data = new_google_database[2000:]

"""
load trained LSTM model
"""
LSTM_model = load_model("trained_model/LSTM_pred.h5")

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = new_google_database.values

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

input_data = new_google_database[len(new_google_database) - len(valid_data_lstm)-60:].values
input_data = scaler.transform(input_data)

LSTM_model.compile(loss = 'mean_squared_error', optimizer = 'adam')
LSTM_model.fit(x_train_data, y_train_data, epochs=1, batch_size=1, verbose=2)

x_test = []
y_test = new_google_database['Close'][2000:]
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

pred_app.layout = html.Div([
    html.H1("Stock Price Analysis Dashtable", style = {"textAlign": "center"}),
    dcc.Tabs(id = "tabs", children=[
        dcc.Tab(label='Company Stock Data', children=[
            html.Div([
                html.H2("Actual Closing Stock Price", style={"textAlign": "center"}),
                dcc.Dropdown(id = 'my-dropdown',
                             options = [{'label': 'Google', 'value': 'GOOGL'},
                                        {'label': 'Tesla', 'value': 'TSLA'},
                                        {'label': 'Apple', 'value': 'AAPL'},
                                        {'label': 'Facebook', 'value': 'FB'},
                                        {'label': 'Microsoft', 'value': 'MSFT'}],
                            multi = True, value = ['GOOGL'],
                            style = {'display': 'block', 'margin-left': 'auto',
                                     'margin-right': 'auto', 'width': '60%'}),
                dcc.Graph(
                    id = "Actual Data",
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
                html.H1("LSTM Predicted Closing Price", style={"textAlign": "center"}),
                dcc.Dropdown(id = 'my-dropdown',
                             options = [{'label': 'Google', 'value': 'GOOGL'},
                                        {'label': 'Tesla', 'value': 'TSLA'},
                                        {'label': 'Apple', 'value': 'AAPL'},
                                        {'label': 'Facebook', 'value': 'FB'},
                                        {'label': 'Microsoft', 'value': 'MSFT'}],
                            multi = True, value = ['GOOGL'],
                            style = {'display': 'block', 'margin-left': 'auto',
                                     'margin-right': 'auto', 'width': '60%'}),
                dcc.Graph(
                    id = "LSTM Predicted Data",
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
        ]),
        dcc.Tab(label = 'Google Stock Data', children = [
            html.Div([
                html.H1('Google Stock High vs Low',
                        style = {'textAlign': 'center'}),
                dcc.Dropdown(id = 'my-dropdown',
                             options = [{'label': 'Google', 'value': 'GOOGL'},
                                        {'label': 'Tesla', 'value': 'TSLA'},
                                        {'label': 'Apple', 'value': 'AAPL'},
                                        {'label': 'Facebook', 'value': 'FB'},
                                        {'label': 'Microsoft', 'value': 'MSFT'}],
                            multi = True, value = ['GOOGL'],
                            style = {'display': 'block', 'margin-left': 'auto',
                                     'margin-right': 'auto', 'width': '60%'}),
                dcc.Graph(id = 'highlow'),
                html.H1("Google Market Volume", style = {'textAlign': 'center'}),
                dcc.Dropdown(id = 'my-dropdown2',
                             options = [{'label': 'Google', 'value': 'GOOGL'},
                                        {'label': 'Tesla', 'value': 'TSLA'},
                                        {'label': 'Apple', 'value': 'AAPL'},
                                        {'label': 'Facebook', 'value': 'FB'},
                                        {'label': 'Microsoft', 'value': 'MSFT'}],
                            multi = True, value = ['GOOGL'],
                            style = {'display': 'block', 'marign-left': 'auto',
                                     'marign-right': 'auto', 'width': '60%'}),
                dcc.Graph(id = 'volume')
            ], className = 'container'),
        ])
    ])
])

if __name__ == '__main__':
    pred_app.run_server(debug = True)