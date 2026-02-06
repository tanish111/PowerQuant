# PowerQuant: GPU Power Consumption Prediction

A complete ML pipeline for predicting GPU power consumption from PyTorch model characteristics using quantile regression.

## 🎯 Quick Start (30 seconds)

```bash
cd /path/to/PowerQuant

# 1. Install dependencies
pip install torch pandas numpy scikit-learn optuna catboost joblib click rich python-dotenv

# 2. Collect dataset (requires NVIDIA GPU)
python build.py collect-dataset

# 3. Train models
python build.py build-model exp05

# 4. Launch web app
cd www && python app.py
# Opens at http://localhost:5000
```

## 🏗️ Project Structure

```
PowerQuant/
├── README.md                                    # This file
├── BUILD_SYSTEM.md                             # Build system documentation
├── SUBSYSTEMS.md                               # Subsystem overview & workflows
├── build.py                                    # Python-based build CLI
├── pyproject.toml                              # Dependencies (uv)
│
├── Dataset Collection/
│   └── Dataset-2(works for Python PyTorch Models)/
│       └── Benchmark Suite/KernelBench/
│           ├── README.md                       # Detailed KernelBench docs
│           ├── scripts/
│           │   ├── generate_baseline_time.py  # Data collection
│           │   ├── extract_model_features.py  # FLOPs & memory extraction
│           │   └── variable_scaler.py         # Feature normalization
│           └── Dataset/                        # Output CSVs (symlinked)
│
├── Model Training/
│   ├── README.md                              # Detailed training docs
│   ├── data/                                  # Symlink to Dataset
│   ├── models/                                # Trained models (joblib)
│   ├── experiments/
│   │   ├── exp01/                            # Baseline experiment
│   │   ├── exp02-05/                         # Alternative configs
│   │   └── exp05/                            # Quantile models (stratified split)
│   └── src/
│       ├── base_classifier/                  # CatBoost, NeuralNet, RandomForest, SVR
│       ├── quantile_classifier/              # Quantile regression wrapper
│       └── utils/                            # Helper utilities
│
├── www/
│   ├── README.md                             # Web app documentation
│   ├── app.py                                # Flask backend
│   ├── index.html                            # CodeMirror editor UI
│   └── models/                               # Symlink to trained models
│
└── Results/
    └── *.tex                                 # Experiment summaries (LaTeX)
```

## 🚀 Main Components

### 1. Build System (`build.py`)
Python-based Makefile-like CLI using Click and `uv` for dependency management.

```bash
python build.py collect-dataset              # Collect GPU benchmark data
python build.py build-model <exp>            # Train models for experiment
python build.py list-experiments             # Show available experiments
python build.py status                       # Check project setup
```

### 2. KernelBench Dataset Collection
PyTorch benchmark suite for extracting GPU kernel features and power measurements.

**Key metrics:**
- FLOPs (via fvcore with 100+ custom op handlers)
- Memory read/write (cumulative model)
- Arithmetic intensity
- Graph structure (nodes, unique ops)
- Power consumption (from PowerAPI or measurements)

**Output:** CSV files with dynamic feature columns

### 3. Model Training Pipeline
Trains 4 quantile regression models on GPU power data.

**Available Models:**
- **CatBoost** - Gradient boosting on categorical features
- **NeuralNet** - Multi-layer perceptron with configurable hidden layers
- **RandomForest** - Ensemble of decision trees
- **SVR** - Support Vector Regression with RBF/polynomial kernels

**Key Features:**
- Optuna hyperparameter tuning (20 trials per model)
- Dynamic feature selection from CSV headers
- Architecture one-hot encoding
- Automatic model serialization with joblib
- Evaluation: R², MAE, RMSE metrics

**Recommended:** Use **exp05** for standard runs (per-architecture split)

### 4. Web Application (Flask) (Under Development)
Interactive web UI for PyTorch model analysis and power prediction.

**Features:**
- CodeMirror editor with syntax highlighting
- Automatic feature extraction from code
- Power predictions from 4 quantile models
- REST API endpoints 
- Model comparison view

**Deploy:** `cd www && python app.py` then open `http://localhost:5000` (Under Development)

## 📊 Data Flow

```
GPU Benchmarks
    ↓
KernelBench (generate_baseline_time.py)
    ├─→ Extract features (FLOPs, memory, intensity)
    ├─→ Measure execution time
    ├─→ Record power consumption
    └─→ Output: combined_*.csv
    ↓
Model Training/data (symlink)
    ↓
Training Pipeline (exp01-05)
    ├─→ Load CSV dynamically
    ├─→ Select numeric columns
    ├─→ One-hot encode architecture
    ├─→ Stratified or configured split
    ├─→ Optuna hyperparameter tuning
    ├─→ Train 4 models
    └─→ Save to models/exp*_quant_*.pkl
```

## 🎯 Experiment Comparison

| Aspect | exp01-05 |
|--------|----------|
| **Dataset** | Per-architecture split |
| **Train/Test** | Stratified |
| **Models** | Base + quantile |
| **Training Time** | Longer |
| **Generalization** | Per-architecture |
| **Scalability** | Limited |

## 📈 Performance Metrics

All models report:
- **R² Score** - Coefficient of determination (range: 0-1, higher = better)
- **MAE** - Mean Absolute Error in watts (lower = better)
- **RMSE** - Root Mean Squared Error in watts (lower = better)

**Example output**:
```
Test R²:   0.854 (explains 85.4% of variance)
Test MAE:  15.23 W (average error)
Test RMSE: 22.15 W (root mean squared error)
```

## 🔗 Feature Names

Automatically detected from CSV headers:

| Feature | Type | Example |
|---------|------|---------|
| `total_flops_m` | Numeric | 1805.3 M |
| `total_bytes_read_mb` | Numeric | 512.5 MB |
| `total_bytes_written_mb` | Numeric | 256.2 MB |
| `arithmetic_intensity` | Numeric | 2.14 FLOP/B |
| `num_nodes` | Numeric | 5 layers |
| `count_unique_ops` | Numeric | 3 types |
| `architecture` | Categorical | K80, Tesla, Ampere |
| `power_consumption` | Target | 185.4 W |

## 🛠️ Common Workflows

### Workflow 1: Full Pipeline (Recommended)
```bash
# Start fresh on new GPU hardware
python build.py collect-dataset              # 1. Collect data
python build.py build-model exp05            # 2. Train models
cd www && python app.py                      # 3. Use predictions
```

### Workflow 2: Retrain with Same Data
```bash
# Retrain exp05 models (data already collected)
python build.py build-model exp05
```

### Workflow 3: Train Single Model
```bash
# Train only CatBoost for exp05
python build.py build-model exp05 --file run_quant_catboost.py
```

### Workflow 4: Compare Experiments
```bash
python build.py build-model exp01              # Baseline
python build.py build-model exp05              # Architecture-stratified
# Compare results in Model Training/experiments/exp*/results.json
```

## 📦 Dependencies

**Core** (auto-installed by `uv sync`):
```
torch>=2.0.0              # Deep learning framework
pandas>=1.5.0             # Data manipulation
numpy>=1.24.0             # Numerical computing
scikit-learn>=1.3.0       # ML utilities
optuna>=3.0.0             # Hyperparameter tuning
catboost>=1.2.0           # One model type
joblib>=1.3.0             # Model serialization
click>=8.1.0              # CLI framework
rich>=13.0.0              # Terminal UI
flask>=2.3.0              # Web framework
python-dotenv>=1.0.0      # Env vars
```

**Optional:**
```
fvcore>=0.1.5             # Advanced FLOP counting
```

**Dev:**
```
pytest>=7.0.0
black>=23.0.0
ruff>=0.1.0
```

## 🔧 Installation

### From Scratch
```bash
# Clone or download project
cd /path/to/PowerQuant

# Install with uv (recommended)
pip install uv
uv sync

# Or install manually
pip install -r requirements.txt
```

### Verify Installation
```bash
python build.py status
# Should show all green checkmarks ✓
```

## 🚢 Deployment

### Local Development
```bash
python build.py build-model exp05
cd www && python app.py
```

### Production (Gunicorn) (Under Development)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 www/app:app
```

### Docker
```bash
# Build
docker build -t powerquant .

# Run
docker run -p 5000:5000 powerquant
```

## ❓ Troubleshooting

### Missing Dependencies
```bash
# Install all required packages
python -m pip install torch pandas numpy scikit-learn optuna catboost joblib click rich python-dotenv flask

# Or use uv
uv sync
```

### "No CUDA GPUs found"
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.device_count())"

# Data collection requires GPU - install NVIDIA drivers
```

### "Port 5000 already in use"
```bash
# Change port in www/app.py
app.run(port=5001)

# Or kill existing process
lsof -ti:5000 | xargs kill -9
```

### "Experiment not found"
```bash
# List available experiments
python build.py list-experiments

# Create exp05 if missing
python build.py build-model exp05
```

### "Models not found for predictions"
```bash
# Check if models are trained
ls Model\ Training/models/

# Should show:
# exp05_quant_catboost.pkl
# exp05_quant_neuralnet.pkl
# exp05_quant_randomforest.pkl
# exp05_quant_svr.pkl

# If missing, retrain
python build.py build-model exp05
```

## 📖 For More Details

- **[BUILD_SYSTEM.md](BUILD_SYSTEM.md)** - Comprehensive build system documentation
- **[SUBSYSTEMS.md](SUBSYSTEMS.md)** - Overview of all subsystems and workflows
- **[Model Training/README.md](Model%20Training/README.md)** - Training pipeline details
- **[KernelBench README](Dataset%20Collection/Dataset-2/Benchmark%20Suite/KernelBench/README.md)** - Data collection

## 🤝 Support

- **Issues & Bugs:** Check [Troubleshooting](#troubleshooting) section
- **Questions:** See subsystem-specific READMEs
- **Suggestions:** Open an issue with feature request

---

**Python:** 3.10+  
**CUDA:** Required for data collection (optional for predictions)
