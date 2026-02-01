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
python build.py build-model exp06

# 4. Launch web app
cd www && python app.py
# Opens at http://localhost:5000
```

## 📚 Documentation

**Start here based on your role:**

- **👤 User/Researcher** → [SUBSYSTEMS.md](SUBSYSTEMS.md)
  - High-level overview of all components
  - Common workflows and examples
  - When to use each tool

- **🛠️ Developer/DevOps** → [BUILD_SYSTEM.md](BUILD_SYSTEM.md)
  - Command-line interface reference
  - Project structure
  - Troubleshooting guide

- **🔬 Data Scientist** → [Model Training/README.md](Model%20Training/README.md)
  - Model architecture and training
  - Hyperparameter tuning with Optuna
  - Experiment comparison

- **📊 Data Engineer** → [Dataset-2 KernelBench README](Dataset%20Collection/Dataset-2/Benchmark%20Suite/KernelBench/README.md)
  - Data collection pipeline
  - Feature extraction (FLOPs, memory)
  - GPU support and benchmarking

- **🌐 Web Developer** → [Web App README](www/README.md)
  - Flask backend and REST API
  - CodeMirror editor integration
  - Model deployment

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
│   │   └── exp06/                            # ⭐ Production model (NEW)
│   │       ├── logger.py                     # Logging
│   │       ├── utils.py                      # Data preparation
│   │       └── run_quant_*.py               # 4 training scripts
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

**Recommended:** Use **exp06** for production (entire dataset, 80/20 random split)

### 4. Web Application (Flask)
Interactive web UI for PyTorch model analysis and power prediction.

**Features:**
- CodeMirror editor with syntax highlighting
- Automatic feature extraction from code
- Power predictions from 4 quantile models
- REST API endpoints
- Model comparison view

**Deploy:** `cd www && python app.py` then open `http://localhost:5000`

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
Training Pipeline (exp06)
    ├─→ Load CSV dynamically
    ├─→ Select numeric columns
    ├─→ One-hot encode architecture
    ├─→ 80/20 random split
    ├─→ Optuna hyperparameter tuning
    ├─→ Train 4 models
    └─→ Save to models/exp06_quant_*.pkl
    ↓
Web Application
    ├─→ Load trained models
    ├─→ Analyze PyTorch code
    ├─→ Extract features (same pipeline)
    └─→ Predict power with quantiles
```

## 🎯 Experiment Comparison

| Aspect | exp01-05 | exp06 (⭐) |
|--------|----------|-----------|
| **Dataset** | Per-architecture split | Entire dataset |
| **Train/Test** | Stratified | Random 80/20 |
| **Models** | 8 (base + quantile) | 4 (quantile only) |
| **Training Time** | Longer | Faster |
| **Generalization** | Per-architecture | Cross-GPU |
| **Scalability** | Limited | Better |
| **Recommended** | Legacy | ✅ Production |

## 📈 Performance Metrics

All models report:
- **R² Score** - Coefficient of determination (range: 0-1, higher = better)
- **MAE** - Mean Absolute Error in watts (lower = better)
- **RMSE** - Root Mean Squared Error in watts (lower = better)

Example output:
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
python build.py build-model exp06            # 2. Train models
cd www && python app.py                      # 3. Use predictions
```

### Workflow 2: Retrain with Same Data
```bash
# Retrain exp06 models (data already collected)
python build.py build-model exp06
```

### Workflow 3: Train Single Model
```bash
# Train only CatBoost for exp06
python build.py build-model exp06 --file run_quant_catboost.py
```

### Workflow 4: Compare Experiments
```bash
python build.py build-model exp01              # Baseline
python build.py build-model exp05              # Architecture-stratified
python build.py build-model exp06              # Random split (recommended)
# Compare results in Model Training/experiments/exp*/results.json
```

### Workflow 5: API Integration
```python
import requests

# Analyze PyTorch model
response = requests.post("http://localhost:5000/api/analyze",
    json={"code": "import torch\n..."})
features = response.json()["features"]

# Get power predictions
response = requests.post("http://localhost:5000/api/predict",
    json={"features": features})
predictions = response.json()["predictions"]

print(predictions["catboost"]["mean"])  # Expected power (W)
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
python build.py build-model exp06
cd www && python app.py
```

### Production (Gunicorn)
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

# Create exp06 if missing
python build.py build-model exp06
```

### "Models not found for predictions"
```bash
# Check if models are trained
ls Model\ Training/models/

# Should show:
# exp06_quant_catboost.pkl
# exp06_quant_neuralnet.pkl
# exp06_quant_randomforest.pkl
# exp06_quant_svr.pkl

# If missing, retrain
python build.py build-model exp06
```

## 📖 For More Details

- **[BUILD_SYSTEM.md](BUILD_SYSTEM.md)** - Comprehensive build system documentation
- **[SUBSYSTEMS.md](SUBSYSTEMS.md)** - Overview of all subsystems and workflows
- **[Model Training/README.md](Model%20Training/README.md)** - Training pipeline details
- **[KernelBench README](Dataset%20Collection/Dataset-2/Benchmark%20Suite/KernelBench/README.md)** - Data collection
- **[Web App README](www/README.md)** - Web interface and API

## 📊 Key Technical Details

### Memory Model
PowerQuant uses **cumulative data movement**, not peak memory:
$$\text{Memory} = \sum_{\text{layers}} (\text{bytes\_read} + \text{bytes\_written})$$

This captures sustained bandwidth usage and is more predictive of power consumption.

### FLOP Counting
Uses `fvcore.nn.FlopCounterMode` with 100+ custom operation handlers:
$$\text{FLOPs} = \sum_{\text{ops}} \text{flops\_per\_op}(W, H, C, K, \ldots)$$

### Arithmetic Intensity
Roofline model metric:
$$I = \frac{\text{FLOPs}}{\text{Bytes Transferred}} \quad [\text{FLOP/Byte}]$$

Higher intensity indicates better compute utilization relative to memory bandwidth.

## 🎓 Citation

If you use PowerQuant in your research, please cite:

```bibtex
@software{powerquant2026,
  title={PowerQuant: GPU Power Consumption Prediction},
  author={Anonymous},
  year={2026},
  url={https://github.com/anonymous/PowerQuant}
}
```

## 📝 License

[Include your license here]

## 👥 Contributing

Contributions welcome! Please:
1. Read [SUBSYSTEMS.md](SUBSYSTEMS.md) for architecture overview
2. Follow code style in existing files
3. Update documentation for new features
4. Test with `python build.py list-experiments`

## 🤝 Support

- **Issues & Bugs:** Check [Troubleshooting](#troubleshooting) section
- **Questions:** See subsystem-specific READMEs
- **Suggestions:** Open an issue with feature request

---

**Last Updated:** February 1, 2026  
**Status:** ✅ Production Ready  
**Python:** 3.10+  
**CUDA:** Required for data collection (optional for predictions)
