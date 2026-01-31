import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import dotenv

# Dynamically find project root (3 levels up from this file)
project_root = str(Path(__file__).resolve().parent.parent.parent)

# Load environment variables if .env exists and override if PROJECT_ROOT is set
dotenv.load_dotenv()
if os.getenv('PROJECT_ROOT'):
    project_root = os.getenv('PROJECT_ROOT')


class ExperimentLogger:
    """Log experiment results in a structured format."""

    def __init__(self, experiment_id: str, results_dir: str = 'results'):
        self.experiment_id = experiment_id
        # Extract model name from experiment_id (e.g., "exp02_catboost" -> "catboost")
        model_name = experiment_id.replace('exp02_', '')
        self.results_dir = Path(f'{project_root}/{results_dir}/exp02/{model_name}/')
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def log_result(
        self,
        hyperparameters: Dict[str, Any],
        metrics: Dict[str, Any],
        test_arch: str,
        dataset_info: Dict[str, Any] = None,
        additional_info: Dict[str, Any] = None,
    ) -> str:
        """Log a single experiment result."""
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime('%Y%m%d_%H%M%S')

        # Create unique hash for this config
        config_str = json.dumps(hyperparameters, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:6]

        result = {
            'metadata': {
                'experiment_id': self.experiment_id,
                'timestamp': timestamp.isoformat(),
                'config_hash': config_hash,
                'test_arch': test_arch,
            },
            'hyperparameters': hyperparameters,
            'results': metrics,
        }

        if dataset_info:
            result['dataset'] = dataset_info

        if additional_info:
            result.update(additional_info)

        # Save to file
        filename = f'{config_hash}_{timestamp_str}.json'
        filepath = self.results_dir / filename

        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)

        return str(filepath)

    def load_all_results(self) -> List[Dict[str, Any]]:
        """Load all results for this experiment."""
        pattern = f'{self.experiment_id}_*.json'
        results = []

        for filepath in self.results_dir.glob(pattern):
            with open(filepath, 'r') as f:
                results.append(json.load(f))

        return results
