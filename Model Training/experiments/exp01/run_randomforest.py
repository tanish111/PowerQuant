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
from src.base_classifier.randomforest import RandomForest, RandomForestConfig


randomforest_config = RandomForestConfig(
    n_estimators=100,
    max_depth=10,
    max_features=1.0,
    min_samples_leaf=10,
    bootstrap=True,
    random_state=42,
    verbose=100,
)


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
    model = RandomForest(config)
    model, metrics = train_and_evaluate_model(model, quantile=False)

    # Log results
    logger = ExperimentLogger(experiment_id='exp01_randomforest')
    logger.log_result(config.to_json_dict(), metrics)

    return metrics['r2']


def main():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)


if __name__ == '__main__':
    main()
