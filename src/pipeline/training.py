import os
import pandas as pd
from termcolor import colored as color
from src.utils import train_test_data


def train_models(data_dict: dict, epochs: int, models: dict, best_params: dict):
    cwd = os.getcwd()
    if os.path.isdir(os.path.join(cwd, 'Models')):
        print("Required data/directory is already present in current working directory.")

    else:
        os.mkdir('Models')

    track = 1

    for ticker in data_dict.keys():
        best_model = best_params[ticker]['Best Model']
        win_size = best_params[ticker]['Window Size']
        seq_type = best_params[ticker]['Sequence']
        lr = best_params[ticker]['Learning Rate']

        data = pd.read_csv(data_dict[ticker])
        X_train, y_train, X_test, y_test = train_test_data(data, split_size=0.2, window_size=win_size, seq=seq_type)
        model = models[best_model](X_train, y_train, lr, model_name=f'{ticker.replace("&", "and")}_Model')

        print(color(f'\n({track}) NSE:{ticker}', 'red', attrs=['bold']))
        print(color('-----------------------', 'yellow'))
        print(color(' >> Model : ', 'blue') + f'{best_model}' + color('\t >> Window Size : ',
                                                                      'blue') + f'{win_size}' + color(
            '\n >> Seq. Type : ', 'blue') + f'{seq_type} Sequence' + color('\t >> Epochs : ', 'blue') + f'{epochs}')

        # fitting model
        model.fit(X_train, y_train, epochs=epochs, batch_size=50, verbose=0)

        testing_error = model.evaluate(X_test, y_test, verbose=0)
        training_error = model.evaluate(X_train, y_train, verbose=0)

        print(
            color(' >> Training Error : ', 'blue') + '{0:.4f}'.format(training_error) + color('\t >> Testing Error : ',
                                                                                              'blue') + '{0:.4f}'.format(
                testing_error))

        # Save the trained model
        model.save(f'Models/{ticker}.h5')

        track += 1

        print(color('\nSuccessfully saved best performing models in \'Models\' directory.', 'magenta'))
