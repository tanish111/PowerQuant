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
from src.base_classifiers.neuralnet import NeuralNet, NeuralNetConfig
from src.quant_classifier import QuantileClassifier


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
        return (8, 100, 1)
    elif key == 'medium':
        return (8, 100, 100, 1)
    else:
        raise ValueError(f'Invalid architecture key: {key}')


def get_objective(test_arch: str):
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

        model, metrics = train_and_evaluate_model(model, test_arch=test_arch, quantile=True)

        # Log results
        logger = ExperimentLogger(experiment_id='exp02_quant_neuralnet')
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
