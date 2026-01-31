import os
import sys

import dotenv

# Set up path before importing from experiments/src
dotenv.load_dotenv()
project_root = os.getenv('PROJECT_ROOT')
sys.path.append(project_root)

# Now import from experiments and src
import optuna

from experiments.exp01.logger import ExperimentLogger
from experiments.exp01.utils import train_and_evaluate_model
from src.base_classifiers.svr import SupportVectorRegressor, SVRConfig
from src.quant_classifier import QuantileClassifier


svr_config = SVRConfig(
    kernel='rbf',
    C=1.0,
    epsilon=0.1,
    gamma='scale',
    degree=3,
    coef0=0.0,
)


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
    base_classifier = SupportVectorRegressor(config)
    model = QuantileClassifier(base_classifier)
    model, metrics = train_and_evaluate_model(model)

    # Log results
    logger = ExperimentLogger(experiment_id='exp01_quant_svr')
    logger.log_result(config.to_json_dict(), metrics)

    return metrics['r2']


def main():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)


if __name__ == '__main__':
    main()
