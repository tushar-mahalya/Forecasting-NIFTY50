import os
import json
import pandas as pd
from collections import OrderedDict
from termcolor import colored as color
from src.utils import train_test_data


def test_models(data_dict, epochs: int, models: dict, window_sizes: list):
    # test result data
    best_params_dict = OrderedDict()
    parameter_dict = OrderedDict()
    sequences = ['Simple', 'Multi']

    for ticker in data_dict.keys():

        lowest_train_error = 1.0
        lowest_test_error = 1.0
        best_model = str()
        best_model_param_count = int()
        best_window_size = int()
        best_seq = str()

        seq_type = {}
        print(color(f'\n================================ NSE : {ticker} ================================', 'red',
                    attrs=["bold"]))
        for seq in sequences:
            print(color('\n--------------------------------------------------------------- ', 'green') + color(
                f'{seq} Sequence', 'green'))

            win_size = {}
            for window_size in window_sizes:

                model_name = {}
                print(color(f"\nWindow size : {window_size}", 'blue'))
                print(color('----------------', 'yellow'))
                for model_type in models.keys():

                    data = pd.read_csv(data_dict[ticker])
                    X_train, y_train, X_test, y_test = train_test_data(data, split_size=0.2, window_size=window_size,
                                                                       seq=seq)
                    model = models[model_type](X_train, y_train, 0.001,
                                               model_name=f'{ticker.replace("&", "and")}_Model')

                    # fitting model
                    model.fit(X_train, y_train, epochs=epochs, batch_size=50, verbose=0)

                    # printing training and testing errors
                    training_error = model.evaluate(X_train, y_train, verbose=0)
                    testing_error = model.evaluate(X_test, y_test, verbose=0)
                    msg = color("  >",
                                'green') + " Model: {0:<15} Param count: {1:} \tTraining error: {2:.4f}\tTesting error: {3:.4f}"
                    print(msg.format(model_type, model.count_params(), training_error, testing_error))

                    model_name[model_type] = {
                        'Param Count': model.count_params(),
                        'Training Error': float("{0:.4f}".format(training_error)),
                        'Testing Error': float("{0:.4f}".format(testing_error))
                    }

                    if testing_error < lowest_test_error:
                        lowest_train_error = training_error
                        lowest_test_error = testing_error
                        best_model = model_type
                        best_model_param_count = model.count_params()
                        best_window_size = window_size
                        best_seq = seq

                win_size[f'Window_Size_{window_size}'] = model_name

            seq_type[f'{seq} Sequence'] = win_size

        parameter_dict[ticker] = seq_type
        best_params_dict[ticker] = {
            'Best Model': best_model,
            'Window Size': best_window_size,
            'Sequence': best_seq,
            'Training Error': lowest_train_error,
            'Testing Error': lowest_test_error,
            'Learning Rate': 0.001,
            'Parameters Count': best_model_param_count
        }

    # Exporting performance of different models and best parameters dict in .json format
    
    with open("artifacts/Testing_Logs.json", "w") as f:
        json.dump(parameter_dict, f)
        f.close()
    with open("artifacts/Best_Model_Params.json", "w") as f:
        json.dump(best_params_dict, f)
        f.close()

    print(color(
        '\nSuccessfully evaluated all data and saved testing logs in \"artifacts/Testing_Logs.json\" and metrics of best performing model at \"artifacts/Best_Model_Params.json\".',
        'magenta'))
