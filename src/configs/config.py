
import os
from src.utils.path import PATH_CONFIG
import src.utils.input_output as io
from src.utils.logger import Logger


class Configs:
    model_variables = list()
    test_set_percent = None
    lag_length = None
    learning_rate = None
    min_child_weight = None
    max_depth = None
    max_delta_step = None
    subsample = None
    colsample_bytree = None
    colsample_bylevel = None
    train_model = None
    model_prediction = None
    feature_importance = None
    make_trading_decisions = None


    def __init__(self, config_file_name: str = None, configs: dict = None):
        """
        Create configs element from file or from a dictionary of values
        Args:
        config_file_name (str): name of the configs file
        configs (dict): settings as a dictionary
        """
        if config_file_name is not None:
            configs = io.read_yaml(PATH_CONFIG, config_file_name)
            Logger.info('Loaded configs from file', os.path.join(PATH_CONFIG, config_file_name),
                        self.__class__.__name__)

        for name, value in configs.items():
            self.__setattr__(name, value)

        # check modeling_steps tasks:
        modeling_steps_tasks = configs.get('modeling_steps')
        for key in modeling_steps_tasks:
            setattr(self, key, modeling_steps_tasks[key])
