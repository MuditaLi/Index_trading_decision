
import numpy as np
import pandas as pd
import src.utils.input_output as io
import matplotlib.pyplot as plt
from os import path
from src.utils.logger import Logger
from src.utils.path import PATH_DATA_OUTPUT, PATH_REPORTS
from sklearn.metrics import mean_squared_error, mean_absolute_error
import shap
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)


def forecast_accuracy(actual, forecast):
    """
    Create model performance measure metrics.
    :param actual: real data
    :param forecast: forecasted data
    :return: MSE, RMSE, MAE values
    """
    mse = mean_squared_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mae = mean_absolute_error(actual, forecast)
    return mse, rmse, mae


def model_predict(model, X_test, y_test):
    """
    Predict on test set and evaluate the prediction.
    :param model: trained model
    :param X_test: Feature test set
    :param y_test: Target test set
    :return: MSE, RMSE, MAE values
    """

    Logger.info('Start predicting on test data')

    y_pred = model.predict(X_test)
    mse, rmse, mae = forecast_accuracy(y_test, y_pred)
    print('Test MSE: %.3f' % mse)
    print('Test MAE: %.3f' % mae)
    print('Test RMSE: %.3f' % rmse)


def feature_importance(model, X_train):
    """
    Generate feature importance values with SHAP package.
    :param model: trained model
    :param X_train: Feature train set
    :return: generate feature importance file
    """

    Logger.info('Start generating feature importance with SHAP')

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap_sum = np.abs(shap_values).mean(axis=0)  # mean along the columns/features
    shap_fi = pd.DataFrame([X_train.columns, shap_sum.tolist()]).T
    shap_fi.columns = ['Feature', 'Score']
    io.write_csv(shap_fi, path.join(PATH_REPORTS, 'feature_importance.csv'))
    # generate feature importance plot
    # shap_plt = shap.summary_plot(shap_values, X_train, plot_type="bar")
    # plot_path = path.join(PATH_DATA_OUTPUT, 'feature_importance_chart.png')
    # shap_plt.savefig(plot_path)


def trading_decision(model, processed_data, idx, lag_length):
    """
    Comparing real and predicted price, then making trading decisions.
    :param model: trained model
    :param processed_data: processed raw data
    :param idx: index data
    :param lag_length: lagging months
    :return: predict vs. real index plot, trading decision
    """

    Logger.info('Start predicting on full data')

    idx.columns = ['Date', 'idx_price']
    idx.index = idx['Date']
    idx.drop(['Date'], axis=1, inplace=True)
    idx = idx.iloc[(lag_length - 1):, ]
    y_pred_full = model.predict(processed_data)
    y_pred_full = pd.DataFrame(y_pred_full)
    output = pd.concat([idx.reset_index(drop=True), y_pred_full], axis=1)
    output.columns = ['real_price', 'predict_price']
    output['real_return'] = (output['real_price'] - output['real_price'].shift(1)) / output['real_price'].shift(1)

    # plot real price and predict price
    plt.plot(idx.index, output['real_price'], label="real_price")
    plt.plot(idx.index, output['predict_price'], label="predict_price")
    plt.title("Equity index test and predicted data")
    plt.legend()
    plot_path = path.join(PATH_REPORTS, 'real_vs_predict.png')
    plt.savefig(plot_path)

    avg_abs_return = np.mean(np.abs(output['real_return'][-12:]))
    sd_abs_return = np.std(np.abs(output['real_return'][-12:]))
    last_return = (y_pred_full.iloc[-1:, 0].values[0]-y_pred_full.iloc[-2:-1, 0].values[0])/y_pred_full.iloc[-2:-1, 0].values[0]
    decision1 = abs(last_return) > avg_abs_return
    decision2 = abs(last_return) > avg_abs_return + sd_abs_return

    print('average absolute return rate(last 12 months): %.3f' % avg_abs_return)
    print('SD absolute return rate(last 12 months): %.3f' % sd_abs_return)
    print('predict last months return rate: %.3f' % last_return)
    print('Is predicted price return larger than the past 12 months mean absolute return?(0/1) %.3f' % decision1)
    print('Is predicted price return larger than the past 12 months mean absolute return + SD?(0/1) %.3f' % decision2)


