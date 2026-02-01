import argparse
import os
import sys

import dotenv

# Set up path before importing from experiments/src
dotenv.load_dotenv()
project_root = os.getenv('PROJECT_ROOT')
sys.path.append(project_root)

# Now import from experiments and src
import optuna

from experiments.exp05.logger import ExperimentLogger
from experiments.exp05.utils import extract_dataset_name, train_and_evaluate_model
from src.base_classifier.svr import SupportVectorRegressor, SVRConfig


svr_config = SVRConfig(
    kernel='rbf',
    C=1.0,
    epsilon=0.1,
    gamma='scale',
    degree=3,
    coef0=0.0,
)


def get_objective(data_path: str, dataset_name: str, test_arch: str):
    def objective(trial):
        # Define hyperparameters
        config = SVRConfig(
            kernel=trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
            C=trial.suggest_float('C', 0.1, 100.0, log=True),
            epsilon=trial.suggest_float('epsilon', 0.01, 1.0, log=True),
            gamma=trial.suggest_categorical('gamma', ['scale', 'auto']),
            degree=trial.suggest_int('degree', 2, 5),
            coef0=trial.suggest_float('coef0', 0.0, 1.0),
        )
        # Train and evaluate
        model = SupportVectorRegressor(config)
        model, metrics = train_and_evaluate_model(model, data_path=data_path, test_arch=test_arch, quantile=False)

        # Log results
        logger = ExperimentLogger(experiment_id='exp05_svr', dataset_name=dataset_name)
        logger.log_result(config.to_json_dict(), metrics, test_arch=test_arch)

        return metrics['r2']

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to dataset CSV (e.g., data/combined_static_20260127_092347.csv)')
    args = parser.parse_args()

    dataset_name = extract_dataset_name(args.dataset)

    # 'a100', 'h100', 'rtx4000', 'rtx5000'
    for test_arch in ['a100', 'h100', 'rtx4000', 'rtx5000']:
        objective = get_objective(args.dataset, dataset_name, test_arch=test_arch)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20)


if __name__ == '__main__':
    main()
