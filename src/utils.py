import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from sklearn.preprocessing import MinMaxScaler


def normalize_feature(stock_df: pd.DataFrame, col: str):
    values = stock_df[col].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(values)
    scaled = scaler.transform(values)
    return list(scaled.flatten()), scaler


def denormalize_feature(scaled_values: list, scaler: MinMaxScaler):
    scaled_values = np.array(scaled_values).reshape(-1, 1)
    denormalized = scaler.inverse_transform(scaled_values)
    return list(denormalized.flatten())


def predictor_response_split(df: pd.DataFrame, window_size: int, seq: str = 'Simple'):
    X = []
    y = []
    if seq == 'Multi':
        numpy_df = df.to_numpy()
        for i in range(len(numpy_df) - window_size):
            row = [r for r in numpy_df[i:i + window_size]]
            X.append(row)
            label = [numpy_df[i + window_size][0]]
            y.append(label)

    elif seq == 'Simple':
        numpy_df = df[['normal_close']].to_numpy()
        for i in range(len(numpy_df) - window_size):
            row = [r for r in numpy_df[i:i + window_size]]
            X.append(row)
            label = numpy_df[i + window_size]
            y.append(label)
    else:
        print('Wrong Choice of sequence.\nType either \'Simple\' or \'Multi\' in \"seq\" attribute.')

    return np.array(X), np.array(y)


def train_test_split(X: np.array, y: np.array, split_size: float):
    train_split = len(X) - int(len(X) * split_size)
    X_train, y_train = X[:train_split], y[:train_split]
    X_test, y_test = X[train_split:], y[train_split:]

    return X_train, y_train, X_test, y_test


def train_test_data(df: pd.DataFrame, split_size: float = 0.2, window_size: int = 7, seq: str = 'Simple'):
    X, y = predictor_response_split(df, window_size, seq)
    X_train, y_train, X_test, y_test = train_test_split(X, y, split_size)

    return X_train, y_train, X_test, y_test


def plot_stock_analysis(stock_df_path: pd.DataFrame, plot_title: str):
    # Loading the data into a pandas DataFrame
    df = pd.read_csv(stock_df_path, parse_dates=['Date'])
    df = df.set_index('Date')
    df = df.drop('Adj Close', axis = 1)
    
    # Plotting Candlestick Chart of the given stock data
    mpf.plot(df, type = 'candle', volume = True, figratio = (15,5), title = f'{plot_title}', style = 'yahoo', tight_layout = True)

    # Calculate the daily returns
    daily_returns = df['Close'].pct_change()

    # Calculate the 30-day rolling standard deviation
    rolling_std = daily_returns.rolling(window=30).std()

    # Calculate the rolling mean of the closing price
    rolling_mean = df['Close'].rolling(window=30).mean()

    # Create the subplots
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))

    # Plot the rolling standard deviation (Volatility Analysis)
    axs[0].plot(rolling_std, color = 'green')
    axs[0].set_title('Volatility Analysis')
    axs[0].set_xlabel('Date')
    axs[0].set_ylabel('Daily Returns Standard Deviation')
    axs[0].tick_params(axis='x', rotation=45)

    # Plot the rolling mean and the closing price (Trend Analysis)
    axs[1].plot(df['Close'][30:], label='Closing Price')
    axs[1].plot(rolling_mean, label='30-Day Rolling Mean')
    axs[1].set_title('Trend Analysis')
    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('Stock Price')
    axs[1].legend()
    axs[1].tick_params(axis='x', rotation=45)

    plt.show()