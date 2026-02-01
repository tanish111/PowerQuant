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


def get_objective(data_path: str, dataset_name: str, test_arch: str):
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
        model, metrics = train_and_evaluate_model(model, data_path=data_path, test_arch=test_arch, quantile=True)

        # Log results
        logger = ExperimentLogger(experiment_id='exp05_quant_randomforest', dataset_name=dataset_name)
        logger.log_result(config.to_json_dict(), metrics, test_arch=test_arch)

        return metrics['r2']

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to dataset CSV (e.g., data/combined_static_20260127_092347.csv)')
    args = parser.parse_args()

    dataset_name = extract_dataset_name(args.dataset)

    # Create models directory
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # 'a100', 'h100', 'rtx4000', 'rtx5000'
    for test_arch in ['a100', 'h100', 'rtx4000', 'rtx5000']:
        print(f"\n{'='*60}")
        print(f"Training for architecture: {test_arch}")
        print(f"{'='*60}")
        
        objective = get_objective(args.dataset, dataset_name, test_arch=test_arch)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20)
        
        # Train final model with best hyperparameters
        print(f"\nBest trial for {test_arch}: {study.best_trial.params}")
        print(f"Best R² score: {study.best_value:.4f}")
        
        best_params = study.best_trial.params
        config = RandomForestConfig(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            min_samples_leaf=best_params['min_samples_leaf'],
        )
        
        base_classifier = RandomForest(config)
        best_model = QuantileClassifier(base_classifier)
        best_model, _ = train_and_evaluate_model(best_model, data_path=args.dataset, test_arch=test_arch)
        
        # Save model
        model_filename = f'exp05_quant_randomforest_{test_arch}_{dataset_name}.pkl'
        model_path = os.path.join(models_dir, model_filename)
        joblib.dump(best_model, model_path)
        print(f"\nModel saved to: {model_path}")


if __name__ == '__main__':
    main()
