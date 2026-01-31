import os
import sys
from pathlib import Path
from typing import Any

import dotenv
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Dynamically find project root (3 levels up from this file)
project_root = str(Path(__file__).resolve().parent.parent.parent)

# Load environment variables if .env exists and override if PROJECT_ROOT is set
dotenv.load_dotenv()
if os.getenv('PROJECT_ROOT'):
    project_root = os.getenv('PROJECT_ROOT')

# lst_selected_features = [
#     'total_bytes_read_mb', 'total_bytes_written_mb',
#     'total_bytes_mb', 'total_flops_m', 'arithmetic_intensity', 'num_nodes',
#     'count_unique_ops', 'architecture', 'power_consumption'
# ]

lst_selected_features = ['total_bytes_read_mb', 'total_bytes_written_mb',
       'total_bytes_mb', 'total_flops_m', 'arithmetic_intensity', 'num_nodes',
       'count_unique_ops', 'memory.total [MiB]', 'memory.reserved [MiB]',
       'protected_memory.total [MiB]', 'power.min_limit [W]',
       'power.max_limit [W]', 'power.default_limit [W]', 'clocks.max.sm [MHz]',
       'clocks.max.memory [MHz]', 'clocks.max.graphics [MHz]',
       'clocks.default_applications.graphics [MHz]',
       'clocks.default_applications.memory [MHz]', 'pcie.link.gen.max',
       'pcie.link.gen.gpumax', 'pcie.link.gen.hostmax', 'pcie.link.width.max',
       'accounting.buffer_size', 'compute_cap', 'architecture',
       'power_consumption']

# lst_selected_features = [
#     'Architecture',
#     'avg_comp_lat',
#     'avg_glob_lat',
#     # 'avg_shar_lat',
#     'glob_inst_kernel',
#     'glob_load_sm',
#     'glob_store_sm',
#     'misc_inst_kernel',
#     'inst_issue_cycles',
#     'cache_penalty',
#     'Avg',
# ]

# lst_selected_features_rename = [
#     'architecture',
#     'avg_comp_lat',
#     'avg_glob_lat',
#     # 'avg_shar_lat',
#     'glob_inst_kernel',
#     'glob_load_sm',
#     'glob_store_sm',
#     'misc_inst_kernel',
#     'inst_issue_cycles',
#     'cache_penalty',
#     'power_consumption',
# ]

lst_selected_features_rename = lst_selected_features


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
    df.columns = lst_selected_features_rename

    y = np.array(df['power_consumption'])

    le = LabelEncoder()
    ind = le.fit_transform(df['architecture'])

    # Remove target and Architecture columns from features
    X = np.array(df.drop(columns=['power_consumption', 'architecture']))

    print(f'Features shape: {X.shape}')
    print(f'Target shape: {y.shape}')

    return X, y, ind


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    ind: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Any, Any]:
    """Split dataframe into training and testing sets."""
    Xtrain, Xtest, ytrain, ytest, indtrain, indtest = train_test_split(
        X, y, ind, test_size=test_size, random_state=random_state
    )

    return (Xtrain, ytrain, indtrain), (Xtest, ytest, indtest)


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
    ind_test: np.ndarray,
    quantile: bool = True,
) -> dict:
    """Evaluate the trained model on test data."""
    if quantile:
        y_pred = model.predict(X_test, ind_test)
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


def train_and_evaluate_model(model: Any, data_path: str, quantile: bool = True):
    # Step 1: Load data
    print('\n1. Loading data...')
    df = load_data(data_path)
    X, y, ind = prepare_data(df)

    # Step 2: Split data
    print('\n2. Splitting data into train/test sets...')
    trainset, testset = split_data(X, y, ind)
    Xtrain, ytrain, indtrain = trainset
    Xtest, ytest, indtest = testset

    # Step 3: Train model
    print('\n3. Training model...')
    model = train_model(Xtrain, ytrain, indtrain, model, quantile=quantile)

    # Step 5: Evaluate model
    print('\n6. Evaluating model...')
    metrics = evaluate_model(model, Xtest, ytest, indtest, quantile=quantile)

    print('\nPipeline completed successfully!')
    return model, metrics


if __name__ == '__main__':
    pass
