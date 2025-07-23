import pandas as pd
import joblib
import optuna
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from config import *
from data_preprocessing import download_and_unzip_data, load_data, preprocess_data
from model import objective, train_final_model
import os
import numpy as np

def run_training():
    """Main function to run the training and hyperparameter optimization."""
    # Download and load data
    if not os.path.exists(os.path.join(RAW_DATA_DIR, 'hour.csv')):
        download_and_unzip_data(DATA_URL, RAW_DATA_DIR)
    
    df = load_data(RAW_DATA_DIR)
    X_train_full, X_test, y_train_full, y_test = preprocess_data(
        df, TARGET_COLUMN, DROP_COLUMNS, TEST_SIZE, RANDOM_STATE
    )

    # Split training data further for validation during hyperparameter tuning
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=RANDOM_STATE
    )
    
    # Identify categorical feature indices for CatBoost
    categorical_features_indices = [X_train.columns.get_loc(col) for col in CATEGORICAL_FEATURES]

    # --- Optuna Hyperparameter Optimization ---
    print("Starting hyperparameter optimization with Optuna...")
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, categorical_features_indices),
        n_trials=OPTUNA_TRIALS,
        timeout=OPTUNA_TIMEOUT
    )

    print(f"Best trial RMSE: {study.best_value}")
    print("Best hyperparameters: ", study.best_params)

    # --- Train Final Model ---
    print("\nTraining final model with best hyperparameters...")
    best_params = study.best_params
    final_model = train_final_model(best_params, X_train_full, y_train_full, categorical_features_indices)
    
    # --- Evaluate Final Model ---
    y_pred = final_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Final Model Evaluation ---")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R^2 Score: {r2:.4f}")

    # --- Save the Model ---
    if not os.path.exists(os.path.dirname(MODEL_PATH)):
        os.makedirs(os.path.dirname(MODEL_PATH))
    joblib.dump(final_model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

if __name__ == "__main__":
    run_training()
