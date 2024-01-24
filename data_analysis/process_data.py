import pandas as pd

"""
preprocessing APPLE stock data
"""
apple_stock = pd.read_csv('data/apple_stock_price.csv')
apple_stock.head()
apple_stock["Date"] = pd.to_datetime(apple_stock.Date, format="%Y-%m-%d")
apple_stock.index = apple_stock['Date']

#create new apple_database for prediction, reformat stock price apple_data into two catalogs
apple_data = apple_stock.sort_index(ascending=True, axis=0)
new_apple_database = pd.DataFrame(index = range(0, len(apple_stock)), columns = ['Date', 'Close'])

for i in range(0, len(apple_data)):
    new_apple_database['Date'][i] = apple_data['Date'][i]
    new_apple_database['Close'][i] = apple_data['Close'][i]

"""
preprocessing FACEBOOK stock data
"""
fackbook_stock = pd.read_csv('data/facebook_stock_price.csv')
fackbook_stock.head()
fackbook_stock["Date"] = pd.to_datetime(fackbook_stock.Date, format="%Y-%m-%d")
fackbook_stock.index = fackbook_stock['Date']

#create new fackbook_stock_database for prediction, reformat stock price fackbook_stock_data into two catalog
facebook_data = fackbook_stock.sort_index(ascending=True, axis=0)
new_facebook_database = pd.DataFrame(index = range(0, len(fackbook_stock)), columns = ['Date', 'Close'])

for i in range(0, len(facebook_data)):
    new_facebook_database['Date'][i] = facebook_data['Date'][i]
    new_facebook_database['Close'][i] = facebook_data['Close'][i]

"""
preprocessing GOOGLE stock data
"""
google_stock = pd.read_csv('data/google_stock_price.csv')
google_stock.head()
google_stock["Date"] = pd.to_datetime(google_stock.Date, format="%Y-%m-%d")
google_stock.index = google_stock['Date']

#create new google_database for prediction, reformat stock price google_data into two catalogs
google_data = google_stock.sort_index(ascending=True, axis=0)
new_google_database = pd.DataFrame(index = range(0, len(google_stock)), columns = ['Date', 'Close'])

for i in range(0, len(google_data)):
    new_google_database['Date'][i] = google_data['Date'][i]
    new_google_database['Close'][i] = google_data['Close'][i]

"""
preprocessing MICROSOFT stock data
"""
microsoft_stock = pd.read_csv('data/microsoft_stock_price.csv')
microsoft_stock.head()
microsoft_stock["Date"] = pd.to_datetime(microsoft_stock.Date, format="%Y-%m-%d")
microsoft_stock.index = microsoft_stock['Date']

#create new microsoft_stock_database for prediction, reformat stock price microsoft_stock_data into two catalogs
microsoft_data = microsoft_stock.sort_index(ascending=True, axis=0)
new_microsoft_database = pd.DataFrame(index = range(0, len(microsoft_stock)), columns = ['Date', 'Close'])

for i in range(0, len(microsoft_data)):
    new_microsoft_database['Date'][i] = microsoft_data['Date'][i]
    new_microsoft_database['Close'][i] = microsoft_data['Close'][i]

"""
preprocessing TESLA stock data
"""
tesla_stock = pd.read_csv('data/tesla_stock_price.csv')
tesla_stock.head()
tesla_stock["Date"] = pd.to_datetime(tesla_stock.Date, format="%Y-%m-%d")
tesla_stock.index = tesla_stock['Date']

#create new fackbook_stock_database for prediction, reformat stock price fackbook_stock_data into two catalogs
tesla_data = tesla_stock.sort_index(ascending=True, axis=0)
new_tesla_database = pd.DataFrame(index = range(0, len(tesla_stock)), columns = ['Date', 'Close'])

for i in range(0, len(facebook_data)):
    new_tesla_database['Date'][i] = tesla_data['Date'][i]
    new_tesla_database['Close'][i] = tesla_data['Close'][i]