import pandas as pd
import numpy as np


def money_flow_index(stock_df: pd.DataFrame, period: int = 14):
    # Fetching Data from Yahoo Finance API
    stock_df = stock_df

    # Calculate typical price
    typical_price = (stock_df['Close'] + stock_df['High'] + stock_df['Low']) / 3
    # typical_price

    # Get the period
    period = period

    # Calculate the money flow
    money_flow = typical_price * stock_df['Volume']

    # Get all the positive and negative money flows

    positive_flow = []
    negative_flow = []

    # Loop through the typical price
    for i in range(1, len(typical_price)):
        if typical_price[i] > typical_price[i - 1]:
            positive_flow.append(money_flow[i - 1])
            negative_flow.append(0)

        elif typical_price[i] < typical_price[i - 1]:
            negative_flow.append(money_flow[i - 1])
            positive_flow.append(0)

        else:
            positive_flow.append(0)
            negative_flow.append(0)

    # Get all the positive and negative money flows within the time period

    positive_mf = []
    negative_mf = []

    for i in range(period - 1, len(positive_flow)):
        positive_mf.append(sum(positive_flow[i + 1 - period: i + 1]))

    for i in range(period - 1, len(negative_flow)):
        negative_mf.append(sum(negative_flow[i + 1 - period: i + 1]))

    # Calculate the money flow index
    MFI = 100 * (np.array(positive_mf) / (np.array(positive_mf) + np.array(negative_mf)))

    # Filling NaN values to starting Period
    mfi_ext = []
    for i in range(period):
        mfi_ext.append(np.NaN)
    mfi_ext.extend(MFI)

    return mfi_ext


def log_return(stock_df: pd.DataFrame):
    # Fetching Data from Yahoo Finance API
    stock_df = stock_df

    log_rtn = np.log(stock_df['Adj Close']) - np.log(stock_df['Adj Close'].shift(1))

    return list(log_rtn)
