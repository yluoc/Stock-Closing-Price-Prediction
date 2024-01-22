import yfinance as yf

stockSymbol = 'GOOGL'

stockData = yf.Ticker(stockSymbol)
stockPrice = stockData.history(period='1d', start='2010-1-1', end='2024-1-1') # google stock price from 2010-1-1 to 2024-1-1

stockPrice.reset_index(inplace=True)
stockPrice.to_csv("data/stock_price.csv", index= False)