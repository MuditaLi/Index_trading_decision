
import pandas as pd
import matplotlib.pyplot as plt
from os import path
from src.utils.logger import Logger
from src.utils.path import PATH_DOCS, PATH_DATA_OUTPUT
import src.utils.input_output as io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
import xgboost as xgb


def prepare_model_data(feature, idx, lag_length):
    idx.columns = ['Date', 'idx_price']
    idx.index = idx['Date']
    idx.drop(['Date'], axis=1, inplace=True)
    # create lagging features - 15 months
    for i in range(1, lag_length):
        idx['lag_' + str(i)] = idx['idx_price'].shift(i)
    # merge feature and index datasets
    dt = feature.merge(idx, how='left', on='Date')
    # change some columns formats
    cols = ['Fed Balance Sheet', 'US Real Personal Income', 'US Real Personal Income exTrans',
            'Adv Retail Sales US exFood Services']
    for i in cols:
        dt[i] = dt[i].apply(lambda x: x.replace(",", ""))
        dt[i] = dt[i].apply(lambda x: x.replace(" ", ""))
        dt[i] = dt[i].apply(lambda x: int(float(x)))
    # move up price to predict
    dt['pred_price'] = dt['idx_price'].shift(-1)
    # truncate dataset
    dt = dt.iloc[(lag_length - 1):, ]
    return dt


def read_model_info():
    Logger.info('Get columns')
    feature_selection = 'feature_selection.yaml'
    features = io.read_yaml(PATH_DOCS, feature_selection)
    model_variables = features['model_variables']
    return model_variables


def dataset_standardization(data):
    """
    Data scaling.
    """

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(data)
    return X_scaled


def split_train_test_xgb(X, y, test_set_percent):
    """
    split data into train set, test set, and last row as forecast set.
    """

    X_forecast = pd.DataFrame(X.iloc[-1, ]).T
    X.drop(X.tail(1).index, inplace=True)
    y.drop(y.tail(1).index, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_set_percent, random_state=1)

    return X_train, y_train, X_test, y_test, X_forecast


def split_train_test_lstm(X, y, test_set_percent):
    """
    Stratified randomized folds preserving the % of samples for each class
    """

    # create train & test sets
    X_train = X[:int(len(X) * (1 - test_set_percent)), :]
    X_test = X[int(len(X) * (1 - test_set_percent)):, :]
    y_train = y.iloc[:int(len(X) * (1 - test_set_percent)), ]
    y_test = y.iloc[int(len(X) * (1 - test_set_percent)):, ]
    # split out last row as forecast data
    X_forecast = X_test[-1, ]
    X_test = X_test[:-1, :]
    y_test = y_test[:-1]
    return X_train, y_train, X_test, y_test, X_forecast


def root_mean_squared_error(y_true, y_pred):
    return tensorflow.keras.backend.sqrt(tensorflow.keras.losses.MSE(y_true, y_pred))


def loss_plot(fitted_model):
    """
    Plot train & test loss plots.
    """

    plt.plot(fitted_model.history["loss"], label="Training Loss")
    plt.plot(fitted_model.history["val_loss"], label="Test Loss")
    plt.title('LSTM model - RMSE')
    plt.legend()
    plot_path = path.join(PATH_DATA_OUTPUT, 'lstm_loss_plot.png')
    plt.savefig(plot_path)


def lstm_model(n_steps, n_features):
    """
    Build up LSTM model structure
    """

    regressor = Sequential()
    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(n_steps, n_features)))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units=1))
    regressor.compile(optimizer='adam', loss=root_mean_squared_error)
    return regressor


def train_lstm_model(feature, idx, save_path, lag_length=16, test_set_percent=0.2, epoch=1000, batch_size=8):
    """
    Train LSTM model
    """

    data = prepare_model_data(feature, idx, lag_length)

    # select variables for training:
    model_variables = read_model_info()
    Logger.info('Start generating LSTM training data')

    # split dataset
    X = pd.DataFrame(data.drop(['Date', 'idx_price', 'pred_price'], axis=1))
    X = X[model_variables].copy()
    y = pd.DataFrame(data['pred_price'])

    # standardize feature data
    X_scaled = dataset_standardization(X)

    # split dataset into test and train
    X_train, y_train, X_test, y_test, X_forecast = split_train_test_lstm(X_scaled, y, test_set_percent)

    # reshape dataset format to fit LSTM model
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    X_forecast = X_forecast.reshape((1, 1, X_forecast.shape[0]))

    # same split datasets into processed folder
    io.write_pkl(X_train, path.join(save_path, 'X_train.pkl'))
    io.write_pkl(y_train, path.join(save_path, 'y_train.pkl'))
    io.write_pkl(X_test, path.join(save_path, 'X_test.pkl'))
    io.write_pkl(y_test, path.join(save_path, 'y_test.pkl'))
    io.write_pkl(X_forecast, path.join(save_path, 'X_forecast.pkl'))

    # build model
    model = lstm_model(X_train.shape[1], X_train.shape[2])

    # fit the LSTM model
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, verbose=2,
                        validation_data=(X_test, y_test), shuffle=True, callback=early_stop)

    file_path = path.join(save_path, 'model.h5')
    model.save(file_path, overwrite=True)

    # save loss plot
    loss_plot(history)


def train_xgb_model(feature, idx, save_path, lag_length=16, test_set_percent=0.2):
    """
    Train XGBoost model
    """

    data = prepare_model_data(feature, idx, lag_length)

    # select variables for training:
    model_variables = read_model_info()
    Logger.info('Start generating XGBoost training data')

    # split dataset
    X = pd.DataFrame(data.drop(['Date', 'idx_price', 'pred_price'], axis=1))
    X = X[model_variables].copy()
    y = pd.DataFrame(data['pred_price'])

    # split dataset into test and train
    X_train, y_train, X_test, y_test, X_forecast = split_train_test_xgb(X, y, test_set_percent)

    # same split datasets into processed folder
    io.write_pkl(X_train, path.join(save_path, 'X_train.pkl'))
    io.write_pkl(y_train, path.join(save_path, 'y_train.pkl'))
    io.write_pkl(X_test, path.join(save_path, 'X_test.pkl'))
    io.write_pkl(y_test, path.join(save_path, 'y_test.pkl'))
    io.write_pkl(X_forecast, path.join(save_path, 'X_forecast.pkl'))

    # fit model
    model = xgb.XGBRegressor(learning_rate=0.03)
    model.fit(X_train, y_train)

    file_path = path.join(save_path, 'model.h5')
    model.save(file_path, overwrite=True)
