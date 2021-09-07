
import os
from pathlib import Path

# Root path
ROOT_NAME = 'multi_asset_cast'
PATH_REPOSITORY_ROOT = Path('.').resolve()

# folder names
DIRNAME_REPORTS = 'reports'
DIRNAME_DATA = 'data'
DIRNAME_SRC = 'src'
DIRNAME_DATA_RAW = 'raw'
DIRNAME_DATA_PROCESSED = 'processed'
DIRNAME_DATA_OUTPUT = 'output'
DIRNAME_CONFIG_FILES = 'configs'
DIRNAME_MODELS = 'models'
DIRNAME_FEATURES = 'features'

# first level path
PATH_REPORTS = os.path.realpath(os.path.join(PATH_REPOSITORY_ROOT, DIRNAME_REPORTS))
PATH_DATA = os.path.realpath(os.path.join(PATH_REPOSITORY_ROOT, DIRNAME_DATA))
PATH_SRC = os.path.realpath(os.path.join(PATH_REPOSITORY_ROOT, DIRNAME_SRC))

# Path to data folder
PATH_DATA_RAW = os.path.realpath(os.path.join(PATH_DATA, DIRNAME_DATA_RAW))
PATH_DATA_OUTPUT = os.path.realpath(os.path.join(PATH_DATA, DIRNAME_DATA_OUTPUT))
PATH_DATA_PROCESSED = os.path.realpath(os.path.join(PATH_DATA, DIRNAME_DATA_PROCESSED))

# Path to configs file folder
PATH_CONFIG = os.path.realpath(os.path.join(PATH_SRC, DIRNAME_CONFIG_FILES))

# Path to models folder
PATH_MODELS = os.path.realpath(os.path.join(PATH_SRC, DIRNAME_MODELS))

# Path to features folder
PATH_FEATURES = os.path.realpath(os.path.join(PATH_SRC, DIRNAME_FEATURES))

PATH_CURRENT_REPOSITORY = os.path.dirname(os.path.realpath(__file__))
