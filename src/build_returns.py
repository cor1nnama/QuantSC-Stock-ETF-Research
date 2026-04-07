import os
import numpy as np
import pandas as pd

DATA_PATH = "data/csv"

def extract_close_returns(file_path: str) -> pd.DataFrame:

    df = pd.read_csv(file_path, header=None)

    tickers = df.iloc[0].tolist()
    fields = df.iloc[1].tolist()

    data = df.iloc[3:].copy()

    data.columns = pd.MultiIndex.from_arrays([tickers, fields])

    date_col = data.columns[0]
    data = data.set_index(date_col)
    data.index.name = "Date"

    close_prices = data.xs("Close", level=1, axis=1)

    close_prices = close_prices.apply(pd.to_numeric, errors="coerce")

    close_prices = close_prices.dropna(axis=1, how="all")

    close_prices = close_prices.loc[:, ~close_prices.columns.duplicated()]

    close_prices.index = pd.to_datetime(close_prices.index, errors="coerce")
    close_prices = close_prices.dropna(axis=0, how="all")
    close_prices = close_prices.sort_index()

    log_prices = np.log(close_prices)
    returns = log_prices.diff().dropna()

    return returns


def load_all_returns(data_path: str = DATA_PATH) -> pd.DataFrame:
    all_returns = []

    for file in os.listdir(data_path):
        if file.endswith(".csv"):
            file_path = os.path.join(data_path, file)
            sector_returns = extract_close_returns(file_path)
            all_returns.append(sector_returns)

    returns = pd.concat(all_returns, axis=1)
    returns = returns.loc[:, ~returns.columns.duplicated()]
    returns = returns.sort_index()

    returns = returns.dropna(how="any")

    return returns