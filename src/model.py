# src/model.py

import optuna
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

def objective(trial, X_train, y_train, X_val, y_val, categorical_features_indices):
    """Optuna objective function."""
    param = {
        'objective': 'RMSE',
        'eval_metric': 'RMSE',
        'iterations': trial.suggest_int('iterations', 500, 2000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'random_strength': trial.suggest_float('random_strength', 1e-9, 10, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 2, 30, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'verbose': False,
        'random_seed': 42
    }

    model = CatBoostRegressor(**param)
    
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              cat_features=categorical_features_indices,
              early_stopping_rounds=50,
              verbose=False)
    
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    
    return rmse

def train_final_model(params, X_train, y_train, categorical_features_indices):
    """Trains the final CatBoost model with the best parameters."""
    model = CatBoostRegressor(**params, verbose=200, random_seed=42)
    model.fit(X_train, y_train, cat_features=categorical_features_indices)
    return model