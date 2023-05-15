from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional
from keras.optimizers import RMSprop


def simple_model(X, y, learn_rate, model_name=None):
    model = Sequential(name=model_name)
    model.add(LSTM(5, input_shape=(X.shape[1:])))
    model.add(Dense(y.shape[1], activation='tanh'))

    # compile the model
    optimizer = RMSprop(learning_rate=learn_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


def dynamic_model(X, y, learn_rate, model_name=None):
    model = Sequential(name=model_name)
    model.add(LSTM(X.shape[1], input_shape=(X.shape[1:])))
    model.add(Dense(y.shape[1], activation='tanh'))

    # compile the model
    optimizer = RMSprop(learning_rate=learn_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


def bidirectional_model(X, y, learn_rate, model_name=None):
    model = Sequential(name=model_name)
    model.add(Bidirectional(LSTM(X.shape[1], return_sequences=False), input_shape=(X.shape[1:])))
    model.add(Dense(X.shape[1]))
    model.add(Dense(y.shape[1], activation='tanh'))

    # compile the model
    optimizer = RMSprop(learning_rate=learn_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


def stacked_model(X, y, learn_rate, model_name=None):
    model = Sequential(name=model_name)
    model.add(LSTM(10, return_sequences=True, input_shape=(X.shape[1:])))
    model.add(LSTM(5))
    model.add(Dense(y.shape[1], activation='tanh'))

    # compile the model
    optimizer = RMSprop(learning_rate=learn_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


def all_models():
    models = {
        "Fixed": simple_model,
        "Dynamic": dynamic_model,
        "Bidirectional": bidirectional_model,
        "Stacked": stacked_model
    }

    return models
