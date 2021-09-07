import numpy as np
import pandas as pd
from os import path
from src.utils.logger import Logger
from src.utils.path import PATH_DATA_OUTPUT, PATH_FEATURES, PATH_DATA_PROCESSED, PATH_REPORTS
import src.utils.input_output as io
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')


def prepare_model_data(feature, idx, lag_length):
    """
    Restructure raw datasets.
    :param feature: feature dataset
    :param idx: index price dataset
    :param lag_length: created index lagging prices as features
    :return: processed data
    """
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

    # generate metrics for report
    dt0 = idx.copy()
    dt0['return'] = (dt0['idx_price']-dt0['idx_price'].shift(1))/dt0['idx_price'].shift(1)
    print(dt0.shape)

    max_month_gain = np.nanmax(dt0['return'])
    max_month_loss = np.nanmin(dt0['return'])
    cum_returns = (1 + dt0['return']).cumprod()
    max_drawdown = np.ptp(cum_returns[1:]) / np.nanmax(cum_returns[1:])
    annual_SR = (np.mean(dt0['return'])*np.sqrt(12))/np.std(dt0['return'])

    def rolling_sharpe(y):
        return np.sqrt(36) * (y.mean() / y.std())
    rolling_SR_3y = dt0['return'].rolling(window=36).apply(rolling_sharpe)
    avg_rolling_3y_SR = np.mean(rolling_SR_3y )

    # generate dataframe
    eval_metrics = pd.DataFrame([])
    eval_metrics['max monthly gain'] = pd.Series([max_month_gain])
    eval_metrics['max monthly loss'] = max_month_loss
    eval_metrics['maximum drawdown'] = max_drawdown
    eval_metrics['annualized Sharpe Ratio'] = annual_SR
    eval_metrics['average rolling 3years SR'] = avg_rolling_3y_SR
    eval_metrics = eval_metrics.T
    eval_metrics.columns = ['value']
    eval_metrics['evaluation metrics'] = eval_metrics.index
    io.write_csv(eval_metrics, path.join(PATH_REPORTS, 'evaluation_metrics.csv'))
    return dt


def read_model_info():
    """
    Select features for the model data.
    :return: feature names to feed model
    """
    Logger.info('Get columns')
    feature_selection = 'feature_selection.yaml'
    features = io.read_yaml(PATH_FEATURES, feature_selection)
    model_variables = features['model_variables']
    return model_variables


def split_train_test_forecast(X, y, test_set_percent, forecast_rows):
    """
    Split data into train set, test set, and last 2 rows as forecast set.
    :param X: feature data
    :param y: target data
    :param test_set_percent: test set percentage
    :param forecast_rows: last rows to forecast on
    :return: X & y split by train, test and forecaset
    """
    X_forecast = pd.DataFrame(X.iloc[-forecast_rows:, ])
    X.drop(X.tail(forecast_rows).index, inplace=True)
    y.drop(y.tail(forecast_rows).index, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_set_percent, random_state=1)

    return X_train, y_train, X_test, y_test, X_forecast


def cv_train(X, y, regressor):
    """
    KFold cross validation training data.
    :param X: feature set
    :param y: target set
    :param regressor: model to train
    :return: trained model
    """

    cv = KFold(n_splits=5, shuffle=True, random_state=1)
    mse = []
    mae = []
    rmse = []
    X = np.array(X)
    y = np.array(y)
    for i, (train, test) in enumerate(cv.split(X, y)):
        regressor.fit(X[train], y[train])
        y_pred = regressor.predict(X[test])
        mse.append(mean_squared_error(y[test], y_pred))
        mae.append(mean_absolute_error(y[test], y_pred))
        rmse.append(np.sqrt(mean_squared_error(y[test], y_pred)))

        print(f'MSE: {np.mean(mse)}')
        print(f'MAE: {np.mean(mae)}')
        print(f'RMSE: {np.mean(rmse)}')
    return regressor


def train_model(feature, idx, save_path, lag_length, test_set_percent, learning_rate, min_child_weight, max_depth,
                max_delta_step, subsample, colsample_bytree, colsample_bylevel):
    """
    Full procedure to train the model.
    """

    data = prepare_model_data(feature, idx, lag_length)

    # select variables for training:
    model_variables = read_model_info()
    Logger.info('Start generating XGBoost training data')

    # split dataset
    X = pd.DataFrame(data.drop(['Date', 'idx_price', 'pred_price'], axis=1))
    X = X[model_variables].copy()
    y = pd.DataFrame(data['pred_price'])

    # save X for later prediction
    io.write_pkl(X, path.join(PATH_DATA_PROCESSED, 'processed_feature_data.pkl'))

    # split dataset into test and train
    X_train, y_train, X_test, y_test, X_forecast = split_train_test_forecast(X, y, test_set_percent, forecast_rows=2)

    # same split datasets into processed folder
    io.write_pkl(X_train, path.join(save_path, 'X_train.pkl'))
    io.write_pkl(y_train, path.join(save_path, 'y_train.pkl'))
    io.write_pkl(X_test, path.join(save_path, 'X_test.pkl'))
    io.write_pkl(y_test, path.join(save_path, 'y_test.pkl'))
    io.write_pkl(X_forecast, path.join(save_path, 'X_forecast.pkl'))

    # fit model
    model = xgb.XGBRegressor(learning_rate=learning_rate,
                             min_child_weight=min_child_weight,
                             max_depth=max_depth,
                             max_delta_step=max_delta_step,
                             subsample=subsample,
                             colsample_bytree=colsample_bytree,
                             colsample_bylevel=colsample_bylevel
                             )
    model = cv_train(X_train, y_train, model)

    # save model
    file_path = path.join(save_path, 'model.pkl')
    pickle.dump(model, open(file_path, "wb"))

    return model



