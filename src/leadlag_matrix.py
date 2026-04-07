import numpy as np
import pandas as pd


def compute_ccf_auc(returns: pd.DataFrame, max_lag: int = 5) -> pd.DataFrame:
    """
    Compute lead-lag matrix S using a ccf-auc style method.

    returns: DataFrame with rows = dates, columns = tickers
    max_lag: number of lags to consider
    """

    tickers = returns.columns
    n = len(tickers)

    S = pd.DataFrame(0.0, index=tickers, columns=tickers)

    for i in range(n):
        for j in range(i + 1, n):
            ti = tickers[i]
            tj = tickers[j]

            x = returns[ti]
            y = returns[tj]

            I_ij = 0.0
            I_ji = 0.0


            for lag in range(1, max_lag + 1):

                corr_ij = x.shift(lag).corr(y)


                corr_ji = y.shift(lag).corr(x)

                if not np.isnan(corr_ij):
                    I_ij += abs(corr_ij)
                if not np.isnan(corr_ji):
                    I_ji += abs(corr_ji)


            if I_ij + I_ji == 0:
                val = 0.0
            else:
                val = np.sign(I_ij - I_ji) * max(I_ij, I_ji) / (I_ij + I_ji)

            S.loc[ti, tj] = val
            S.loc[tj, ti] = -val

    return S