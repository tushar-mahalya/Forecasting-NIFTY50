import os
import json
import pickle
import shutil
import yfinance as yf

from src.components.features import money_flow_index, log_return
from src.utils import normalize_feature


def stock_data(ticker: str):
    ticker = ticker
    stock_df = yf.download(ticker + '.NS',
                           start='2022-01-01',
                           end='2022-12-31',
                           progress=False)
    return stock_df.reset_index()


def data_accumilator(src_location: str, stocks_lst: list):
    preprocessed_data_file = {}

    root_dir = os.path.join(src_location, 'data')
    art_dir = os.path.join(src_location, 'artifacts')
    
    os.mkdir(f'{src_location}/artifacts/Scaler Objects')
    
    for ticker in stocks_lst:
        stock_df = stock_data(ticker)
        norm_stock = stock_df.copy()

        # Calculating Money-Flow Index and Logarithmic Return
        # for the underlying stock
        mfi = money_flow_index(stock_df, 14)
        returns = log_return(stock_df)

        norm_stock['MFI'] = mfi
        norm_stock['Returns'] = returns

        # Normalizing Values
        norm_close, scaler_obj = normalize_feature(norm_stock, 'Adj Close')
        norm_mfi, _ = normalize_feature(norm_stock, 'MFI')
        norm_returns, _ = normalize_feature(norm_stock, 'Returns')

        norm_stock['normal_close'] = norm_close
        norm_stock['normal_mfi'] = norm_mfi
        norm_stock['normal_returns'] = norm_returns

        # Finalizing Pre-processed Data
        norm_stock = norm_stock[['Date', 'normal_close', 'normal_mfi', 'normal_returns']][14:]
        norm_stock.set_index('Date', drop=True, inplace=True)

        stock_df.to_csv(f'{ticker}.csv', index=False)
        norm_stock.to_csv(f'Normalized_{ticker}.csv', index=False)

        folder_path = os.path.join(src_location, ticker)
        simple_file_path = os.path.join(src_location, f'{ticker}.csv')
        normalized_file_path = os.path.join(src_location, f'Normalized_{ticker}.csv')

        os.mkdir(folder_path)
        shutil.move(simple_file_path, folder_path)
        shutil.move(normalized_file_path, folder_path)
        shutil.move(folder_path, root_dir)

        # saving scaler object for reverting back the normalized values
        with open(f'{src_location}/artifacts/Scaler Objects/{ticker}_scaler.pkl', 'wb') as f:
            pickle.dump(scaler_obj, f)

        orignal_data_path = os.path.join(src_location, 'data', ticker, f'{ticker}.csv')
        norm_data_path = os.path.join(src_location, 'data', ticker, f'Normalized_{ticker}.csv')
        scaler_obj_path = os.path.join(src_location, 'artifacts', 'Scaler Objects', f'{ticker}_scaler.pkl')
        
        preprocessed_data_file[ticker] = {
                                          'Orignal Data' : orignal_data_path,
                                          'Normalized Data' : norm_data_path, 
                                          'Scaler Object' : scaler_obj_path
                                         }

    # Writing data mapping into JSON format
    with open(f'{art_dir}/data_map.json', 'w') as file:
        json.dump(preprocessed_data_file, file)

    print(f"Successfully downloaded the OHLCV data from source, created new features, normalized and saved in \'data\' directory.\n Also saved the scaler objects and data mapping file at \'artifacts\' directory.\n\nData directory path -> {root_dir}\nArtifacts directory path -> {art_dir}")
