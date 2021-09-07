
import logging
import os.path
import pickle
import yaml
import pandas as pd

logging.basicConfig(
    # filename=os.path.join(PATH_OUTPUT, 'import.log'),
    format='%(levelname)s %(asctime)s %(message)s',
    level=logging.DEBUG
)


def construct_path(*path):
    path = os.path.join(*path)
    path = os.path.join(os.path.dirname(__file__), path) if not os.path.isabs(path) else path
    return path


def model_prediction(path):
    pass


def read_hdf5(*path):
    """
    Creates a dictionary from a YALM file
    """
    # Read YAML file
    path = construct_path(*path)
    return model_prediction(path)


def read_yaml(*path):
    """
    Creates a dictionary from a YALM file
    """
    # Read YAML file
    path = construct_path(*path)
    with open(path, 'r') as stream:
        data_loaded = yaml.load(stream, Loader=yaml.FullLoader)
    return data_loaded


def write_yaml(dictionary, *path):
    """
    Write dictionary as a YAML file
    """
    path = construct_path(*path)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w', encoding='utf8') as outfile:
        yaml.dump(dictionary, outfile, default_flow_style=False, allow_unicode=True)
    return path


def write_pkl(obj, *path):
    """
    Filepath in .pkl
    """
    path = construct_path(*path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def read_pkl(*path):
    path = construct_path(*path)
    with open(path, 'rb') as f:
        return pickle.load(f)


def read_csv(*path):
    path = construct_path(*path)
    return pd.read_csv(path)


def write_csv(dataframe: pd.DataFrame, *path, **kwargs):
    """
    Writes a DataFrame to disk as csv file, maintaining project standards.
    """
    path = construct_path(*path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dataframe.to_csv(path, index=False, **kwargs)


def write_xlsx(dataframe: pd.DataFrame, *path, **kwargs):
    """
    Writes a DataFrame to disk as xlsx file, maintaining project standards.
    """
    path = construct_path(*path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dataframe.to_excel(path, index=False, **kwargs)
