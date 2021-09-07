
from src.utils.path import PATH_DATA_RAW, PATH_DATA_PROCESSED
import src.utils.input_output as io


class DataLoader:

    # downloaded raw filenames
    FILE_NAME_FEATURE = 'multiasset_feature.csv'
    FILE_NAME_INDEX = 'multiasset_index.csv'
    FILE_NAME_PROCESSED_RAW = 'processed_feature_data.pkl'

    # processed filenames
    FILE_NAME_X_TRAIN = 'X_train.pkl'
    FILE_NAME_X_TEST = 'X_test.pkl'
    FILE_NAME_Y_TRAIN = 'y_train.pkl'
    FILE_NAME_Y_TEST = 'y_test.pkl'
    FILE_NAME_X_FORECAST = 'X_forecast.pkl'

    # output filenames
    FILE_NAME_MODEL = 'model.pkl'

    def __init__(self):
        """
        This object deals with all data loading tasks
        """

    def load_raw_data_feature(self):
        return io.read_csv(PATH_DATA_RAW, self.FILE_NAME_FEATURE)

    def load_raw_data_index(self):
        return io.read_csv(PATH_DATA_RAW, self.FILE_NAME_INDEX)

    def load_processed_raw_data(self):
        return io.read_pkl(PATH_DATA_PROCESSED, self.FILE_NAME_PROCESSED_RAW)

    def load_model(self):
        return io.read_pkl(PATH_DATA_PROCESSED, self.FILE_NAME_MODEL)

    def load_X_train(self):
        return io.read_pkl(PATH_DATA_PROCESSED, self.FILE_NAME_X_TRAIN)

    def load_X_test(self):
        return io.read_pkl(PATH_DATA_PROCESSED, self.FILE_NAME_X_TEST)

    def load_y_train(self):
        return io.read_pkl(PATH_DATA_PROCESSED, self.FILE_NAME_Y_TRAIN)

    def load_y_test(self):
        return io.read_pkl(PATH_DATA_PROCESSED, self.FILE_NAME_Y_TEST)

    def load_X_forecast(self):
        return io.read_pkl(PATH_DATA_PROCESSED, self.FILE_NAME_X_FORECAST)