# PowerQuant Web Application

A web-based tool for analyzing PyTorch models and predicting power consumption using machine learning models.

## Features

- 🎨 **Interactive Code Editor**: Write PyTorch models with syntax highlighting and validation
- 📊 **Model Analysis**: Extract features like FLOPs, memory usage, and computational intensity
- ⚡ **Power Prediction**: Estimate power consumption using 4 quantile regression models:
  - CatBoost
  - Neural Network
  - Random Forest
  - Support Vector Regressor
- 📝 **Templates**: Pre-built templates for common model architectures
- ✅ **Code Validation**: Real-time Python syntax checking

## Installation

1. **Install dependencies:**
   ```bash
   cd www
   pip install -r requirements.txt
   ```

2. **Install additional dependencies** (if not already installed):
   ```bash
   # For feature extraction
   pip install fvcore

   # For model training (if using pre-trained models)
   pip install catboost scikit-learn
   ```

## Running the Application

1. **Start the Flask server:**
   ```bash
   python app.py
   ```

2. **Open your browser and navigate to:**
   ```
   http://localhost:5000
   ```

## Usage

### 1. Write Your Model

Follow the KernelBench template structure:

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    """Your model description"""
    def __init__(self, param1, param2):
        super(Model, self).__init__()
        # Define layers
        self.layer = nn.Conv2d(param1, param2, 3)

    def forward(self, x):
        # Define forward pass
        x = self.layer(x)
        return x

# Global variables
batch_size = 64
param1 = 64
param2 = 128

def get_inputs():
    # Return sample input tensors
    return [torch.rand(batch_size, param1, 224, 224)]

def get_init_inputs():
    # Return model initialization parameters
    return [param1, param2]
```

### 2. Analyze the Model

Click **"Analyze & Predict"** to:
- Extract model features (FLOPs, memory usage, etc.)
- Get power consumption predictions from all 4 models
- View results with confidence intervals (lower, median, upper bounds)

### 3. Load Templates

Use the template dropdown to load pre-built examples:
- Conv2d + ReLU
- Conv2d + GELU
- Matrix Multiplication
- Scaled Dot Product Attention

## API Endpoints

### GET `/api/templates`
Get available model templates.

### POST `/api/validate`
Validate Python code syntax.
```json
{
  "code": "import torch\n..."
}
```

### POST `/api/analyze`
Analyze model and extract features.
```json
{
  "code": "import torch\n..."
}
```

**Response:**
```json
{
  "success": true,
  "features": {
    "total_bytes_read_mb": 123.45,
    "total_bytes_written_mb": 67.89,
    "total_flops_m": 1234.56,
    "arithmetic_intensity": 10.0,
    "num_nodes": 15,
    "count_unique_ops": 8
  }
}
```

### POST `/api/predict`
Predict power consumption from features.
```json
{
  "features": {
    "total_bytes_read_mb": 123.45,
    ...
  }
}
```

**Response:**
```json
{
  "success": true,
  "predictions": {
    "catboost": {
      "lower": 85.23,
      "median": 120.45,
      "upper": 155.67
    },
    ...
  }
}
```

## Project Structure

```
www/
├── app.py                 # Flask backend
├── templates/
│   └── index.html        # Frontend UI
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Integrating Trained Models

To use your actual trained models instead of mock predictions:

1. **Load your trained models in `app.py`:**

```python
# Load pre-trained models
catboost_model = CatBoost.load('path/to/catboost_model.pkl')
neuralnet_model = NeuralNet.load('path/to/neuralnet_model.pth')
# ... etc
```

2. **Update the `/api/predict` endpoint:**

```python
@app.route('/api/predict', methods=['POST'])
def predict_power():
    features = request.get_json()['features']
    X = prepare_features(features)  # Convert to model input format
    
    predictions = {
        'catboost': {
            'lower': catboost_model.predict_quantile(X, 0.25),
            'median': catboost_model.predict(X),
            'upper': catboost_model.predict_quantile(X, 0.75)
        },
        # ... repeat for other models
    }
    
    return jsonify({'success': True, 'predictions': predictions})
```

## Troubleshooting

### Feature extraction not working
- Ensure `fvcore` is installed: `pip install fvcore`
- Check that the KernelBench path in `app.py` is correct

### Models not available
- Verify that Model Training modules are importable
- Check Python path configuration in `app.py`

### Port already in use
- Change the port in `app.py`: `app.run(port=5001)`

## Development

To enable debug mode and auto-reload:
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

## License

MIT License - see parent project for details

## Credits

Built on top of PowerQuant and KernelBench projects.
