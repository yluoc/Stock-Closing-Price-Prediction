import yfinance as yf

google_stockSymbol = 'GOOGL'
tesla_stockSymbol = 'TSLA'
facebook_stockSymbol = 'META'
microsoft_stockSymbol = 'MSFT'
apple_stockSymbol = 'AAPL'


stockData1 = yf.Ticker(google_stockSymbol)
stockPrice1 = stockData1.history(period='1d', start='2010-1-1', end='2024-1-1') # google stock price from 2010-1-1 to 2024-1-1
stockPrice1.reset_index(inplace=True)
stockPrice1.to_csv("data/google_stock_price.csv", index= False)

stockData2 = yf.Ticker(tesla_stockSymbol)
stockPrice2 = stockData2.history(period='1d', start='2010-1-1', end='2024-1-1')
stockPrice2.reset_index(inplace=True)
stockPrice2.to_csv("data/tesla_stock_price.csv", index= False)

stockData3 = yf.Ticker(facebook_stockSymbol)
stockPrice3 = stockData3.history(period='1d', start='2010-1-1', end='2024-1-1')
stockPrice3.reset_index(inplace=True)
stockPrice3.to_csv("data/facebook_stock_price.csv", index= False)

stockData4 = yf.Ticker(microsoft_stockSymbol)
stockPrice4 = stockData4.history(period='1d', start='2010-1-1', end='2024-1-1')
stockPrice4.reset_index(inplace=True)
stockPrice4.to_csv("data/microsoft_stock_price.csv", index= False)

stockData5 = yf.Ticker(apple_stockSymbol)
stockPrice5 = stockData5.history(period='1d', start='2010-1-1', end='2024-1-1')
stockPrice5.reset_index(inplace=True)
stockPrice5.to_csv("data/apple_stock_price.csv", index= False)