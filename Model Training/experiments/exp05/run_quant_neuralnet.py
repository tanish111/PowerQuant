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
from src.base_classifier.neuralnet import NeuralNet, NeuralNetConfig
from src.quantile_classifier.quant_classifier import QuantileClassifier


neuralnet_config = NeuralNetConfig(
    hidden_layer_sizes=(8, 100, 1),
    activation='relu',
    optimizer='adam',
    learning_rate=0.001,
    momentum=0.9,
    num_epochs=1000,
    batch_size=128,
    loss_function='mse',
    weight_decay=0.0001,
)


def get_arch(key: str):
    if key == 'small':
        return (7, 100, 1)
    elif key == 'medium':
        return (7, 100, 100, 1)
    else:
        raise ValueError(f'Invalid architecture key: {key}')


def get_objective(data_path: str, dataset_name: str, test_arch: str):
    def objective(trial):
        # Define hyperparameters
        config = NeuralNetConfig(
            hidden_layer_sizes=get_arch(
                trial.suggest_categorical('hidden_layer_sizes', ['small', 'medium'])
            ),
            activation=trial.suggest_categorical('activation', ['relu']),
            optimizer=trial.suggest_categorical('optimizer', ['adam', 'sgd']),
            learning_rate=trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            momentum=trial.suggest_float('momentum', 0.7, 0.9),
            num_epochs=trial.suggest_categorical('num_epochs', [1000, 2000, 5000]),
            batch_size=trial.suggest_categorical('batch_size', [256, 512, 1024]),
            loss_function=trial.suggest_categorical('loss_function', ['mse', 'bce']),
            weight_decay=trial.suggest_float('weight_decay', 0.0001, 0.01, log=True),
        )

        # Train and evaluate
        base_classifier = NeuralNet(config)
        model = QuantileClassifier(base_classifier)

        model, metrics = train_and_evaluate_model(model, data_path=data_path, test_arch=test_arch, quantile=True)

        # Log results
        logger = ExperimentLogger(experiment_id='exp05_quant_neuralnet', dataset_name=dataset_name)
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
        config = NeuralNetConfig(
            hidden_layer_sizes=best_params['hidden_layer_sizes'],
            activation=best_params['activation'],
            dropout_rate=best_params['dropout_rate'],
            learning_rate=best_params['learning_rate'],
            num_epochs=best_params['num_epochs'],
            batch_size=best_params['batch_size'],
        )
        
        base_classifier = NeuralNet(config)
        best_model = QuantileClassifier(base_classifier)
        best_model, _ = train_and_evaluate_model(best_model, data_path=args.dataset, test_arch=test_arch)
        
        # Save model
        model_filename = f'exp05_quant_neuralnet_{test_arch}_{dataset_name}.pkl'
        model_path = os.path.join(models_dir, model_filename)
        joblib.dump(best_model, model_path)
        print(f"\nModel saved to: {model_path}")


if __name__ == '__main__':
    main()
