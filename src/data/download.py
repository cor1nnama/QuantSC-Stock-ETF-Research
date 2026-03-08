# scripts to download sources
# TODO: Kevin + Karan work on this

import yfinance as yf

# Dictionary of stock tickers (grouped by sector)

tickers = {"Tech": ["AAPL", "MSFT", "GOOG", "NVDA", "ADBE", "ORCL", "INTC", "TXN", "AMD", "PYPL"], 
           "Financials": ["JPM", "BAC", "C", "GS", "MS", "BRK-A", "V", "MA", "AXP", "BLK"], 
           "Consumer_Disc": ["TSLA", "NKE", "HD", "LOW", "SBUX", "MCD", "TGT", "TJX", "LULU", "SONO"], 
           "Industrials": ["GE", "CAT", "BA", "MMM", "HON", "ITW", "UNP", "LMT", "FDX", "RTX"], 
           "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL", "SRE", "PEG", "CNP"]}


# Dictionary to store downloaded data for each ticker
# - start="2018-01-01" → fetch data from Jan 1st, 2018
# - group_by="ticker" → group data by stock ticker

data = {}

for key in tickers.keys():
    data.update({key: yf.download(tickers[key], start="2018-01-01", group_by="ticker")})