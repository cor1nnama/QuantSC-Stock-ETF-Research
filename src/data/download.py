# scripts to download sources
# TODO: Kevin + Karan work on this

import yfinance as yf

# Dictionary of stock tickers (grouped by sector)

tickers = {"Tech": ["AAPL", "MSFT", "GOOG", "NVDA", "ADBE", "ORCL", "INTC", "TXN", "AMD", "PYPL"], 
           "Financials": ["JPM", "BAC", "C", "GS", "MS", "BRK-A", "V", "MA", "AXP", "BLK"], 
           "Consumer_Disc": ["TSLA", "NKE", "HD", "LOW", "SBUX", "MCD", "TGT", "TJX", "LULU", "SONO"], 
           "Industrials": ["GE", "CAT", "BA", "MMM", "HON", "ITW", "UNP", "LMT", "FDX", "RTX"], 
           "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL", "SRE", "PEG", "CNP"],
           "Healthcare": ["LLY", "JNJ", "ABBV", "MRK", "PFE", "AMGN", "GILD", "ISRG", "HCA", "UNH"],
           "Energy": ["XOM","CVX","COP","EOG","SLB","OXY","DVN","FANG","LNG","HAL"],
           "Consumer_Staples": ["PG","KO","PEP","WMT","COST","CL","KMB","GIS","HSY","MKC"],
           "Real_Estate": ["AMT","PLD","EQIX","O","SPG","WELL","DLR","AVB","EQR","VICI"],
           "Communication_Services": ["META","NFLX","DIS","TMUS","VZ","T","CHTR","FOXA","CMCSA","GOOG"]
          }


# Dictionary to store downloaded data for each ticker
# - start="2018-01-01" → fetch data from Jan 1st, 2018
# - group_by="ticker" → group data by stock ticker

data = {}

for sector, sector_tickets in tickers.items():
    print("Downloading:", sector)
    data.update({key: yf.download(sector_tickers, start="2018-01-01", group_by="ticker")})
