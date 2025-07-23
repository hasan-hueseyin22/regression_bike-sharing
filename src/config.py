# Paths
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"
RAW_DATA_DIR = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/"
MODEL_PATH = "models/catboost_model.joblib"

# Data columns
# 'dteday' will be parsed, 'instant' is just an index
DROP_COLUMNS = ['instant', 'dteday', 'casual', 'registered']
TARGET_COLUMN = 'cnt'

# Features are all columns except the dropped ones and the target
# CatBoost can handle categorical features internally
CATEGORICAL_FEATURES = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']

# Model & Training settings
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Optuna Hyperparameter Optimization
OPTUNA_TRIALS = 50
OPTUNA_TIMEOUT = 300  # in seconds
