import pandas as pd
from sklearn.model_selection import train_test_split
import requests
import zipfile
import io
import os

def download_and_unzip_data(url, save_path):
    """Downloads and unzips data from a URL."""
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print("Downloading data...")
    response = requests.get(url)
    response.raise_for_status()
    
    zip_file = zipfile.ZipFile(io.BytesIO(response.content))
    zip_file.extractall(save_path)
    print(f"Data downloaded and extracted to {save_path}")

def load_data(path):
    """Loads and preprocesses data from the csv file."""
    df = pd.read_csv(os.path.join(path, 'hour.csv'))
    # No need to parse 'dteday' as 'yr', 'mnth', etc. are already extracted
    return df

def preprocess_data(df, target_column, drop_columns, test_size, random_state):
    """Drops unnecessary columns and splits data."""
    df = df.drop(columns=drop_columns)
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Convert categorical features to category dtype for CatBoost
    for col in X.columns:
        if X[col].dtype == 'object' or len(X[col].unique()) < 25: # Heuristic for categorical
             X[col] = X[col].astype('category')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test
