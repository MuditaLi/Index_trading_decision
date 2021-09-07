
import sys
import time
from src.utils.logger import Logger
from src.data.data_loader import DataLoader
from src.configs.config import Configs
from src.models.train_model import train_model
from src.models.predict_model import model_predict, feature_importance, trading_decision
from src.utils.path import PATH_DATA_PROCESSED


def main(configs: Configs = None, data_loader: DataLoader = None):
    """
    Load configs file to select configures and functions to run.
    :param configs: use yaml file to select configures
    :param data_loader: loading dataset function
    """

    if configs is None:
        configs = Configs('config_function_selection.yaml')

    if data_loader is None:
        data_loader = DataLoader()

    if configs.train_model:
        raw_feature = data_loader.load_raw_data_feature()
        raw_index = data_loader.load_raw_data_index()
        train_model(feature=raw_feature, idx=raw_index, save_path=PATH_DATA_PROCESSED, lag_length=configs.lag_length,
                    test_set_percent=configs.test_set_percent, learning_rate=configs.learning_rate,
                    min_child_weight=configs.min_child_weight, max_depth=configs.max_depth,
                    max_delta_step=configs.max_delta_step, subsample=configs.subsample,
                    colsample_bytree=configs.colsample_bytree, colsample_bylevel=configs.colsample_bylevel)

    if configs.model_prediction:
        model = data_loader.load_model()
        X_test = data_loader.load_X_test()
        y_test = data_loader.load_y_test()
        model_predict(model, X_test, y_test)

    if configs.feature_importance:
        model = data_loader.load_model()
        X_train = data_loader.load_X_train()
        feature_importance(model, X_train)

    if configs.make_trading_decisions:
        processed_data = data_loader.load_processed_raw_data()
        raw_index = data_loader.load_raw_data_index()
        model = data_loader.load_model()
        trading_decision(model, processed_data, raw_index, lag_length=configs.lag_length)


if __name__ == '__main__':
    START = time.time()

    configs = None
    if len(sys.argv) > 1:
        configs = Configs(sys.argv[1])

    main(configs)
    END = time.time()
    Logger.info('Script completed in', '%i seconds' % (END - START), __file__)
