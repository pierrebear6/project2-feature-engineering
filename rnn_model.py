from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import *
from data import *
from kerastuner.tuners import RandomSearch
from datetime import datetime
from kerastuner.engine.hyperparameters import HyperParameters
from data import *


# MODEL
def model(window_size, n_features, output_size):
    model = Sequential()
    model.add(LSTM(units=160, return_sequences=True, input_shape=(window_size, n_features)))
    model.add(LSTM(units=192, return_sequences=True))
    model.add(LSTM(units=16, return_sequences=False))
    model.add(Dense(24))
    model.add(Dense(units=output_size))
    model.compile(loss='mae', optimizer='adam')
    model.summary()
    print(model)

    return model


# COMPILE
def model_fit(model, X_train, y_train, X_val, y_val):
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=10,
                                   restore_best_weights=True)
    result = model.fit(X_train,
                       y_train,
                       epochs=100,
                       batch_size=32,
                       shuffle=False,
                       validation_data=(X_val, y_val),
                       callbacks=[early_stopping],
                       verbose=1)

    mae = pd.DataFrame(result.history).rename(columns={'loss': 'Training',
                                                       'val_loss': 'Validation'})

    return model, mae


def rnn_model_predict(model, unscaled_df, labels, X_test, window_size, unscaled_test, n_future, original_df):
    pred_labels = []
    unscaled_test = unscaled_test[window_size + n_future:]

    for i in labels:
        pred_labels.append('pred_' + i)

    predictions = pd.DataFrame(model.predict(X_test), columns=pred_labels)
    predictions = PrepareData(df1=unscaled_df[labels], ticker=None).inverse_scale_data(scaled_data=predictions)
    forecast = predictions[-n_future:]
    predictions = predictions[:-n_future]
    actuals = unscaled_test[labels]
    model_df = predictions
    for i in labels:
        model_df[i] = actuals[i].values

    model_df['Date'] = unscaled_test['Date'].values
    model_df.set_index('Date', inplace=True)
    model_df['Date'] = model_df.index
    model_df['Date'] = pd.to_datetime(model_df['Date'])

    return model_df, forecast


def tuned_model(window_size, n_features, output_size, X_train, y_train, X_val, y_val):
    LOG_DIR = f"{filepath}/data/tuned_rnn_models/{str(datetime.now().strftime('%d-%m-%Y %H-%M-%S'))}"

    def build_model(hp):
        model = Sequential()
        # Input
        model.add(LSTM(units=hp.Int('input_units', min_value=16, max_value=256, step=16), return_sequences=True,
                       input_shape=(window_size, n_features)))

        # Hidden
        for i in range(hp.Int('n_layers', 1, 3)):
            model.add(LSTM(hp.Int(f'lstm_{i}_units', min_value=16, max_value=256, step=16), return_sequences=True))

        model.add(LSTM(hp.Int('translate_units', min_value=8, max_value=128, step=8), return_sequences=False))

        for i in range(hp.Int('n_dense_layers', 0, 1)):
            model.add(Dense(hp.Int(f'dense_{i}_units', min_value=8, max_value=64, step=8)))

        # Output
        model.add(Dense(units=output_size))

        # Compile
        model.compile(loss='mae', optimizer=hp.Choice('optimizer', ['adam', 'RMSProp']))
        model.summary()

        return model

    tuner = RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=100,  # how many model variations to test?
        executions_per_trial=2,  # how many trials per variation? (same model could perform differently)
        directory=LOG_DIR)

    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=10,
                                   restore_best_weights=True)

    tuner.search(x=X_train,
                 y=y_train,
                 verbose=1,  # just slapping this here bc jupyter notebook. The console out was getting messy.
                 epochs=100,
                 batch_size=32,
                 callbacks=[early_stopping],  # if you have callbacks like tensorboard, they go here.
                 validation_data=(X_val, y_val))
    best_hps = tuner.get_best_hyperparameters()[0].values
    best_model = tuner.get_best_models()[0]
    best_model_summary = tuner.get_best_models()[0].summary()
    print(best_hps)
    print(best_model_summary)
    return best_hps, best_model

