import pandas as pd
import matplotlib.pyplot as plt

sd = pd.read_csv('data/stock_price.csv')
sd.head()
sd["Date"] = pd.to_datetime(sd.Date, format="%Y-%m-%d")
sd.index = sd['Date']
"""
create new database for prediction, reformat stock price data into two catalogs
"""
data = sd.sort_index(ascending=True, axis=0)
new_database = pd.DataFrame(index = range(0, len(sd)), columns = ['Date', 'Close'])

for i in range(0, len(data)):
    new_database['Date'][i] = data['Date'][i]
    new_database['Close'][i] = data['Close'][i]
