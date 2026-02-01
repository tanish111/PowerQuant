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
from src.base_classifier.randomforest import RandomForest, RandomForestConfig
from src.quantile_classifier.quant_classifier import QuantileClassifier


randomforest_config = RandomForestConfig(
    n_estimators=100,
    max_depth=10,
    max_features=1.0,
    min_samples_leaf=10,
    bootstrap=True,
    random_state=42,
    verbose=100,
)


def get_objective(test_arch: str):
    def objective(trial):
        # Define hyperparameters
        config = RandomForestConfig(
            n_estimators=trial.suggest_int('n_estimators', 500, 5000),
            max_depth=trial.suggest_int('max_depth', 1, 10),
            max_features=trial.suggest_float('max_features', 0.1, 1.0),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
            bootstrap=trial.suggest_categorical('bootstrap', [True, False]),
            random_state=42,
            verbose=0,
        )

        # Train and evaluate
        base_classifier = RandomForest(config)
        model = QuantileClassifier(base_classifier)
        # test_arch = 'ampere'  # Options: "adalovelace", "ampere", "k80", "tesla"
        model, metrics = train_and_evaluate_model(model, test_arch=test_arch, quantile=True)

        # Log results
        logger = ExperimentLogger(experiment_id='exp02_quant_randomforest')
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
