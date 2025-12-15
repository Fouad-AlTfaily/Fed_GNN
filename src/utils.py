"""
Utility functions for FedGATSage implementation
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import logging
import json
import os
import time
from typing import Dict, List, Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )

def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seeds set to {seed}")

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     class_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics"""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, balanced_accuracy_score
    
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Overall metrics
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Create detailed report
    report = {
        'accuracy': float(accuracy),
        'balanced_accuracy': float(balanced_acc),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'per_class': {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'support': support.tolist()
        }
    }
    
    if class_names:
        report['class_names'] = class_names
        report['per_class_detailed'] = {
            class_names[i]: {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            }
            for i in range(len(class_names))
        }
    
    return report

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: Optional[List[str]] = None,
                         save_path: Optional[str] = None) -> plt.Figure:
    """Plot and optionally save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    return fig

def plot_training_progress(losses: List[float], round_times: List[float],
                          save_path: Optional[str] = None) -> plt.Figure:
    """Plot training progress over federation rounds"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    rounds = list(range(1, len(losses) + 1))
    ax1.plot(rounds, losses, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Federation Round')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Progress')
    ax1.grid(True, alpha=0.3)
    
    # Plot round times
    ax2.plot(rounds, round_times, 'r-s', linewidth=2, markersize=6)
    ax2.set_xlabel('Federation Round')
    ax2.set_ylabel('Round Time (seconds)')
    ax2.set_title('Training Time per Round')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training progress plot saved to {save_path}")
    
    return fig

def save_results(results: Dict[str, Any], save_path: str):
    """Save experimental results to JSON file"""
    # Convert numpy arrays and tensors to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif isinstance(value, torch.Tensor):
            serializable_results[key] = value.cpu().numpy().tolist()
        else:
            serializable_results[key] = value
    
    with open(save_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Results saved to {save_path}")

def load_dataset_info(data_dir: str) -> Dict[str, Any]:
    """Load dataset information and statistics"""
    info = {
        'detector_types': [],
        'client_counts': {},
        'attack_distributions': {}
    }
    
    # Check for detector directories
    detector_types = ['temporal', 'content', 'behavioral']
    for detector_type in detector_types:
        detector_dir = os.path.join(data_dir, f'{detector_type}_detector')
        if os.path.exists(detector_dir):
            info['detector_types'].append(detector_type)
            
            # Count clients
            client_files = [f for f in os.listdir(detector_dir) if f.startswith('client_') and f.endswith('.csv')]
            info['client_counts'][detector_type] = len(client_files)
            
            # Analyze attack distribution in test file
            test_path = os.path.join(detector_dir, 'test.csv')
            if os.path.exists(test_path):
                df = pd.read_csv(test_path)
                attack_counts = df['Attack'].value_counts()
                info['attack_distributions'][detector_type] = attack_counts.to_dict()
    
    logger.info(f"Dataset info loaded: {len(info['detector_types'])} detector types found")
    return info

def validate_model_consistency(model_states: List[Dict[str, torch.Tensor]]) -> bool:
    """Validate that all client models have consistent architecture"""
    if not model_states:
        return False
    
    reference_keys = set(model_states[0].keys())
    reference_shapes = {key: model_states[0][key].shape for key in reference_keys}
    
    for i, state in enumerate(model_states[1:], 1):
        current_keys = set(state.keys())
        if current_keys != reference_keys:
            logger.error(f"Model {i} has inconsistent keys")
            return False
        
        for key in reference_keys:
            if state[key].shape != reference_shapes[key]:
                logger.error(f"Model {i} has inconsistent shape for {key}")
                return False
    
    logger.info("All models have consistent architecture")
    return True

class ExperimentTracker:
    """Track and log experimental progress"""
    
    def __init__(self, experiment_name: str, save_dir: str):
        self.experiment_name = experiment_name
        self.save_dir = save_dir
        self.metrics_history = defaultdict(list)
        self.start_time = None
        
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Experiment tracker initialized: {experiment_name}")
    
    def start_experiment(self):
        """Mark experiment start time"""
        self.start_time = time.time()
        logger.info(f"Experiment '{self.experiment_name}' started")
    
    def log_round_metrics(self, round_idx: int, metrics: Dict[str, Any]):
        """Log metrics for a federation round"""
        self.metrics_history['round'].append(round_idx)
        for key, value in metrics.items():
            self.metrics_history[key].append(value)
        
        logger.info(f"Round {round_idx} metrics logged")
    
    def save_experiment(self, final_results: Optional[Dict[str, Any]] = None):
        """Save complete experiment results"""
        results = {
            'experiment_name': self.experiment_name,
            'total_time': time.time() - self.start_time if self.start_time else 0,
            'metrics_history': dict(self.metrics_history),
            'final_results': final_results or {}
        }
        
        save_path = os.path.join(self.save_dir, f'{self.experiment_name}_results.json')
        save_results(results, save_path)
        
        return results