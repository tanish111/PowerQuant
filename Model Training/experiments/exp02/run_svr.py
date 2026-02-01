import os
import sys

import dotenv

# Set up path before importing from experiments/src
dotenv.load_dotenv()
project_root = os.getenv('PROJECT_ROOT')
sys.path.append(project_root)

# Now import from experiments and src
import optuna

from experiments.exp02.logger import ExperimentLogger
from experiments.exp02.utils import train_and_evaluate_model
from src.base_classifier.svr import SupportVectorRegressor, SVRConfig


svr_config = SVRConfig(
    kernel='rbf',
    C=1.0,
    epsilon=0.1,
    gamma='scale',
    degree=3,
    coef0=0.0,
)


def get_objective(test_arch: str):
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
        # test_arch = 'ampere'  # Options: "adalovelace", "ampere", "k80", "tesla"
        model, metrics = train_and_evaluate_model(model, test_arch=test_arch, quantile=False)

        # Log results
        logger = ExperimentLogger(experiment_id='exp02_svr')
        logger.log_result(config.to_json_dict(), metrics, test_arch=test_arch)

        return metrics['r2']

    return objective


def main():
    objective = get_objective(test_arch='ampere')
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)

    objective = get_objective(test_arch='k80')
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)

    objective = get_objective(test_arch='tesla')
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)

    objective = get_objective(test_arch='adalovelace')
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)


if __name__ == '__main__':
    main()
