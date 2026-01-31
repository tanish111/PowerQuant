import os
import sys
from pathlib import Path
from typing import Any

import dotenv
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Dynamically find project root (3 levels up from this file)
project_root = str(Path(__file__).resolve().parent.parent.parent)

# Load environment variables if .env exists and override if PROJECT_ROOT is set
dotenv.load_dotenv()
if os.getenv('PROJECT_ROOT'):
    project_root = os.getenv('PROJECT_ROOT')

lst_selected_features = [
    'total_bytes_read_mb', 'total_bytes_written_mb',
    'total_bytes_mb', 'total_flops_m', 'arithmetic_intensity', 'num_nodes',
    'count_unique_ops', 'architecture', 'power_consumption'
]


def extract_dataset_name(data_path: str) -> str:
    """Extract dataset name from path like 'data/combined_static_20260127_092347.csv'.

    Returns: 'static_20260127_092347'
    """
    filename = os.path.basename(data_path)  # combined_static_20260127_092347.csv
    name = filename.replace('combined_', '').replace('.csv', '')
    return name  # static_20260127_092347


def load_data(data_path: str) -> pd.DataFrame:
    """Load the preprocessed GPU kernel performance data.

    Args:
        data_path: Path to the dataset CSV file (required).
    """
    full_path = os.path.join(project_root, data_path)
    df = pd.read_csv(full_path)
    print(f'Data loaded successfully. Shape: {df.shape}')
    print(f'Columns: {list(df.columns)}')
    return df


def prepare_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare features and target variable from the dataset."""
    # Set target variable
    df = df[lst_selected_features]
    y = np.array(df['power_consumption'])

    le = LabelEncoder()
    ind = le.fit_transform(df['architecture'])

    # Remove target and Architecture columns from features
    X = np.array(df.drop(columns=['power_consumption', 'architecture']))

    print(f'Features shape: {X.shape}')
    print(f'Target shape: {y.shape}')

    return X, y, ind


# For experiment 05, the data is split by architecture.
def split_data(
    df: pd.DataFrame,
    test_arch: str = 'a100',  #'a100', 'h100', 'rtx4000', 'rtx5000'
) -> tuple[Any, Any]:
    """Split dataframe into training and testing sets."""
    df_test = df[df['architecture'] == test_arch]
    df_train = df[df['architecture'] != test_arch]
    print(f'Training set size: {df_train.shape[0]} samples')
    print(f'Test set size: {df_test.shape[0]} samples')
    return df_train, df_test


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    ind_train: np.ndarray,
    model: Any,
    quantile: bool = True,
) -> Any:
    """Train a model."""
    if quantile:
        model.fit(X_train, y_train, ind_train)
    else:
        model.fit(X_train, y_train)

    return model


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    val_test: np.ndarray,
    quantile: bool = True,
) -> dict:
    """Evaluate the trained model on test data."""
    if quantile:
        y_pred = model.predict(X_test, val=val_test)
    else:
        y_pred = model.predict(X_test)

    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
    }

    print('\n' + '=' * 50)
    print('MODEL EVALUATION RESULTS')
    print('=' * 50)
    print(f'Root Mean Squared Error (RMSE): {metrics["rmse"]:.4f}')
    print(f'Mean Absolute Error (MAE): {metrics["mae"]:.4f}')
    print(f'R² Score: {metrics["r2"]:.4f}')
    print('=' * 50)

    return metrics


def train_and_evaluate_model(model: Any, data_path: str, test_arch: str = 'a100', quantile: bool = True):
    # Step 1: Load data
    print('\n1. Loading data...')
    df = load_data(data_path)
    df_train, df_test = split_data(df, test_arch=test_arch)

    # Step 2: Prepare data
    print('\n2. Preparing data...')
    X_train, y_train, ind_train = prepare_data(df_train)
    X_test, y_test, _ = prepare_data(df_test)

    # This is the distribution of the power values.
    # np.sort is to verify no test leakage while prediction.
    val_test = np.sort(df_test['power_consumption'])

    # Step 3: Train model
    print('\n3. Training model...')
    model = train_model(X_train, y_train, ind_train, model, quantile=quantile)

    # Step 5: Evaluate model
    print('\n6. Evaluating model...')
    metrics = evaluate_model(model, X_test, y_test, val_test, quantile=quantile)

    print('\nPipeline completed successfully!')
    return model, metrics


if __name__ == '__main__':
    pass
