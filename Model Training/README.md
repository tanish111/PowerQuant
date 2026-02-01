# Model Training Subsystem

Model Training is the core ML pipeline for training quantile regression models to predict GPU power consumption.

## Overview

The subsystem implements an ensemble of 4 quantile regression models:
1. **CatBoost** - Gradient boosting on categorical features
2. **NeuralNet** - Multi-layer perceptron with configurable hidden layers
3. **RandomForest** - Ensemble of decision trees
4. **SVR** - Support Vector Regression with RBF/polynomial kernels

Each model is trained on GPU kernel performance data and wrapped in a `QuantileClassifier` for ensemble predictions.

## Directory Structure

```
Model Training/
├── data/                          # Symlink to Dataset Collection/Dataset-2/Dataset
├── models/                        # Trained model storage (joblib format)
├── experiments/
│   ├── exp01/                    # Baseline experiment (stratified split)
│   ├── exp02/                    # Alternative configuration
│   ├── exp04/                    # Another variant
│   ├── exp05/                    # Quantile with 3+1 architecture split
│   └── exp06/                    # **NEW** Quantile with random 80/20 split
├── main.py                        # Entry point (if used)
└── src/
    ├── base_classifier/          # Core model implementations
    │   ├── catboost.py
    │   ├── neuralnet.py
    │   ├── randomforest.py
    │   ├── svr.py
    │   └── xgboost.py
    ├── quantile_classifier/       # Quantile regression wrapper
    │   └── quant_classifier.py
    └── utils/
        ├── clean_datasets.py
        ├── scribble.py
        └── summarize_results.py
```

## Experiments

### Exp01-05: Legacy Experiments
- Various configurations with different split strategies
- Include both base models and quantile wrappers
- Useful for comparison and ablation studies

### **Exp06: Production Model** (Recommended)

The newest experiment optimized for production deployment:

```bash
# Train all 4 quantile models
python ../build.py build-model exp06

# Train specific model
python ../build.py build-model exp06 --file run_quant_catboost.py
```

**Key Features:**
- ✅ Uses **entire dataset** (no per-architecture partitioning)
- ✅ **Random 80/20 train/test split** for better generalization
- ✅ **Hyperparameter tuning** with Optuna (20 trials per model)
- ✅ **Automatic model saving** to `models/exp06_quant_*.pkl`
- ✅ **Dynamic feature selection** from CSV headers
- ✅ **Architecture one-hot encoding** for categorical handling

## Training a Model

### Using the Build System (Recommended)

```bash
cd /path/to/PowerQuant
python build.py build-model exp06
```

### Running Directly

```bash
cd Model\ Training
export PROJECT_ROOT=$(pwd)/..
python experiments/exp06/run_quant_catboost.py --dataset data/combined_static_20260127_092347.csv
```

## Data Preparation

The `prepare_data()` function in each experiment's `utils.py`:

1. **Loads CSV** with dynamic column detection
2. **Filters to numeric columns** only (excludes string model names)
3. **Removes target** (`power_consumption`) from features
4. **One-hot encodes** architecture (if present)
5. **Fills missing values** with appropriate defaults

Example CSV columns detected:
```
- total_bytes_read_mb      (numeric)
- total_bytes_written_mb   (numeric)
- total_flops_m            (numeric)
- arithmetic_intensity     (numeric)
- num_nodes                (numeric)
- count_unique_ops         (numeric)
- architecture             (categorical → one-hot encoded)
- power_consumption        (target)
```

## Model Architecture

### Base Model (e.g., CatBoost)

```python
from src.base_classifier.catboost import CatBoost, CatBoostConfig

config = CatBoostConfig(
    iterations=500,
    learning_rate=0.01,
    depth=6,
    loss_function='RMSE',
)
model = CatBoost(config)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Quantile Wrapper

```python
from src.quantile_classifier.quant_classifier import QuantileClassifier

quant_model = QuantileClassifier(
    base_model=model,
    model_type="catboost",
    architecture_encoder=X_arch
)

# Returns quantile predictions
pred_dict = quant_model.predict(X_test)
# {
#   "mean": [...],
#   "lower_q": [...],      # e.g., 25th percentile
#   "upper_q": [...]       # e.g., 75th percentile
# }
```

## Hyperparameter Tuning

Each model uses **Optuna** for hyperparameter optimization:

```python
sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction='minimize', sampler=sampler)
study.optimize(objective_fn, n_trials=20)
```

Tuning objectives:
- **CatBoost**: iterations, learning_rate, depth, l2_leaf_reg
- **NeuralNet**: hidden_layer_size, learning_rate, max_iter, batch_size
- **RandomForest**: n_estimators, max_depth, min_samples_split/leaf
- **SVR**: kernel, C, epsilon, degree

## Evaluation Metrics

All models report:
- **R² Score** - Coefficient of determination (train & test)
- **MAE** - Mean Absolute Error (train & test)
- **RMSE** - Root Mean Squared Error (train & test)

Example output:
```
✓ Test R²:  0.8542
✓ Test MAE: 15.23 W
✓ Test RMSE: 22.15 W
```

## Model Saving

Trained models are serialized with `joblib`:

```python
import joblib

# Save
joblib.dump(model, "models/exp06_quant_catboost.pkl")

# Load
model = joblib.load("models/exp06_quant_catboost.pkl")
predictions = model.predict(X_new)
```

## Integration with Web App

The `www/app.py` Flask backend loads trained models at startup:

```python
import joblib

# Load all exp06 models
models = {
    "catboost": joblib.load("Model Training/models/exp06_quant_catboost.pkl"),
    "neuralnet": joblib.load("Model Training/models/exp06_quant_neuralnet.pkl"),
    "randomforest": joblib.load("Model Training/models/exp06_quant_randomforest.pkl"),
    "svr": joblib.load("Model Training/models/exp06_quant_svr.pkl"),
}

# Use for predictions
pred = models["catboost"].predict(X_test)
```

## Troubleshooting

### Memory Error During Training
- Reduce batch_size in NeuralNet config
- Use smaller dataset or subset
- Enable virtual memory swap

### Slow Hyperparameter Tuning
- Reduce n_trials in Optuna study
- Use faster samplers (RandomSampler instead of TPESampler)
- Parallelize with `n_jobs=-1`

### Feature Mismatch at Prediction Time
- Ensure input features match training features
- Use same preprocessing pipeline
- Verify architecture encoding matches

### Missing Dataset
```bash
# Check data symlink
python build.py status --verbose

# Manually create if needed
ln -s ../Dataset\ Collection/Dataset-2/Dataset Model\ Training/data
```

## Advanced Usage

### Custom Dataset
```bash
python build.py build-model exp06 --dataset data/my_custom.csv
```

### Retraining on New Data
```bash
# Collect new baseline data
python build.py collect-dataset

# Retrain models
python build.py build-model exp06
```

### Model Comparison
```bash
python build.py list-experiments
# Try different experiments to compare strategies
python build.py build-model exp01
python build.py build-model exp05
python build.py build-model exp06
```

## Configuration Files

Each experiment stores its configuration in:
- `experiments/exp0X/logger.py` - Logging utilities
- `experiments/exp0X/utils.py` - Data loading & preprocessing
- `experiments/exp0X/run_*.py` - Model training scripts

## Related Documentation

- [Build System](../BUILD_SYSTEM.md) - How to run training
- [KernelBench README](../Dataset%20Collection/Dataset-2/Benchmark%20Suite/KernelBench/README.md) - Data collection
- [Web App README](../www/README.md) - Model deployment
