import os
import sys

import dotenv

# Set up path before importing from experiments/src
dotenv.load_dotenv()
project_root = os.getenv('PROJECT_ROOT')
sys.path.append(project_root)

# Now import from experiments and src
import numpy as np
import optuna

from experiments.exp01.logger import ExperimentLogger
from experiments.exp01.utils import train_and_evaluate_model
from src.base_classifier.catboost import CatBoost, CatBoostConfig
from src.quantile_classifier.quant_classifier import QuantileClassifier


# Global variable for dataset path
DATASET_PATH = None

catboost_config = CatBoostConfig(
    iterations=5000,
    learning_rate=0.01,
    depth=1,
    random_seed=42,
    verbose=500,
    loss_function='RMSE',
    eval_metric='RMSE',
    l2_leaf_reg=0.1,
)


class LoglossObjective(object):
    def calc_ders_range(self, approxes, targets, weights):
        assert len(approxes) == len(targets)
        if weights is not None:
            raise ValueError('Weights are not supported')
        result = []
        y = np.clip(targets, 1e-3, 1 - 1e-3)
        p = np.clip(approxes, 1e-3, 1 - 1e-3)
        der1 = (y - p) / (p * (1 - p))
        der2 = -y / p**2 - (1 - y) / (1 - p) ** 2
        result = [(x, y) for x, y in zip(der1, der2)]
        return result


def objective(trial):
    # Define hyperparameters
    config = CatBoostConfig(
        depth=trial.suggest_int('depth', 1, 10),
        learning_rate=trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        iterations=trial.suggest_categorical('iterations', [1000, 2000, 5000]),
        min_data_in_leaf=trial.suggest_int('min_data_in_leaf', 10, 100),
        l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 0.1, 10.0, log=True),
        loss_function=trial.suggest_categorical('loss_function', ['RMSE', 'MAE', 'Logloss']),
    )
    if config.loss_function == 'Logloss':
        config.loss_function = LoglossObjective()

    # Train and evaluate
    base_classifier = CatBoost(config)
    model = QuantileClassifier(base_classifier)
    model, metrics = train_and_evaluate_model(model, data_path=DATASET_PATH)

    # Log results
    logger = ExperimentLogger(experiment_id='exp01_quant_catboost')
    logger.log_result(config.to_json_dict(), metrics)

    return metrics['r2']


def main():
    global DATASET_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='data/combined_static_20260127_092347.csv')
    args = parser.parse_args()
    DATASET_PATH = args.dataset
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)


if __name__ == '__main__':
    main()
