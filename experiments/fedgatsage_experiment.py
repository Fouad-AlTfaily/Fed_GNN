"""
Main experiment script for FedGATSage.
Demonstrates the complete pipeline from data loading to evaluation.
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from federated_learning import FedGATSageSystem
from utils import (setup_logging, set_random_seeds, calculate_metrics, 
                   plot_confusion_matrix, plot_training_progress, save_results,
                   load_dataset_info, ExperimentTracker)
from community_detection import CommunityAwareProcessor

import logging
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='FedGATSage Experiment')
    
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--dataset', type=str, choices=['nf_ton_iot', 'cic_ton_iot'],
                       default='cic_ton_iot', help='Dataset to use')
    parser.add_argument('--num_clients', type=int, default=5,
                       help='Number of federated clients')
    parser.add_argument('--num_rounds', type=int, default=15,
                       help='Number of federation rounds')
    parser.add_argument('--detector_types', nargs='+', 
                       default=['temporal', 'content', 'behavioral'],
                       help='Detector types to use')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--demo_mode', action='store_true',
                       help='Run in demo mode (reduced complexity)')
    
    return parser.parse_args()

def setup_experiment(args):
    """Setup experiment environment"""
    # Setup logging
    log_file = os.path.join(args.output_dir, 'experiment.log')
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.log_level, log_file)
    
    # Set random seeds
    set_random_seeds(args.seed)
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    logger.info(f"Experiment arguments: {vars(args)}")
    
    return device

def demonstrate_community_abstraction(data_dir: str):
    """Demonstrate the community abstraction mechanism"""
    logger.info("=== DEMONSTRATING COMMUNITY ABSTRACTION ===")
    
    processor = CommunityAwareProcessor()
    print(processor.explain_flow_as_community_abstraction())
    
    # Try to load a sample dataset to show community detection
    detector_types = ['temporal', 'content', 'behavioral']
    for detector_type in detector_types:
        detector_dir = os.path.join(data_dir, f'{detector_type}_detector')
        sample_file = os.path.join(detector_dir, 'client_1.csv')
        
        if os.path.exists(sample_file):
            logger.info(f"Demonstrating community detection on {detector_type} detector data")
            
            # Load small sample
            df = pd.read_csv(sample_file).head(1000)  # Just first 1000 rows
            
            # Check required columns
            if 'Src IP' in df.columns and 'Dst IP' in df.columns:
                try:
                    df_enhanced = processor.create_community_enhanced_features(df, {})
                    
                    if processor.communities:
                        num_communities = len(set(processor.communities.values()))
                        logger.info(f"Found {num_communities} communities in sample data")
                        
                        # Show community distribution
                        community_sizes = pd.Series(processor.communities).value_counts()
                        logger.info(f"Community sizes: {dict(community_sizes.head())}")
                    
                    break  # Successfully demonstrated, exit loop
                    
                except Exception as e:
                    logger.warning(f"Could not demonstrate community detection on {detector_type}: {e}")
                    continue
    
    logger.info("=== COMMUNITY ABSTRACTION DEMONSTRATION COMPLETE ===")

def run_federated_experiment(args, device: str) -> dict:
    """Run the main federated learning experiment"""
    logger.info("Starting FedGATSage federated learning experiment")
    
    # Initialize experiment tracker
    experiment_name = f"fedgatsage_{args.dataset}_{args.num_clients}clients_{args.num_rounds}rounds"
    tracker = ExperimentTracker(experiment_name, args.output_dir)
    tracker.start_experiment()
    
    # Load dataset information
    dataset_info = load_dataset_info(args.data_dir)
    logger.info(f"Dataset info: {dataset_info}")
    
    # Adjust parameters for demo mode
    if args.demo_mode:
        args.num_rounds = min(args.num_rounds, 5)
        logger.info("Running in demo mode with reduced rounds")
    
    # Initialize FedGATSage system
    fed_system = FedGATSageSystem(
        data_dir=args.data_dir,
        num_clients=args.num_clients,
        detector_types=args.detector_types,
        device=device
    )
    
    # Determine model dimensions based on available data
    # Try to load a sample to get input dimensions
    sample_loader = None
    input_dim = 64  # Default fallback
    
    for detector_type in args.detector_types:
        detector_dir = os.path.join(args.data_dir, f'{detector_type}_detector')
        if os.path.exists(detector_dir):
            from federated_learning import DataLoader
            sample_loader = DataLoader(detector_dir, detector_type)
            sample_data = sample_loader.load_client_data(1)
            if sample_data and 'features' in sample_data:
                input_dim = sample_data['features'].shape[1]
                break
    
    # Get number of classes
    num_classes = 8  # Default for IoT datasets
    if sample_loader and sample_loader.label_mapper:
        num_classes = len(sample_loader.label_mapper)
    
    logger.info(f"Model configuration: input_dim={input_dim}, num_classes={num_classes}")
    
    # Initialize models
    fed_system.initialize_models(
        input_dim=input_dim,
        hidden_dim=256,
        num_classes=num_classes
    )
    
    # Run federated training
    training_results = fed_system.train_federated(num_rounds=args.num_rounds)
    
    # Evaluate final performance
    evaluation_results = evaluate_system(fed_system, args)
    
    # Combine results
    final_results = {
        'training': training_results,
        'evaluation': evaluation_results,
        'configuration': {
            'num_clients': args.num_clients,
            'num_rounds': args.num_rounds,
            'detector_types': args.detector_types,
            'input_dim': input_dim,
            'num_classes': num_classes
        }
    }
    
    # Log final metrics
    if evaluation_results:
        tracker.log_round_metrics(args.num_rounds, {
            'final_accuracy': evaluation_results.get('accuracy', 0.0),
            'final_f1': evaluation_results.get('macro_f1', 0.0)
        })
    
    # Save experiment
    tracker.save_experiment(final_results)
    
    return final_results

def evaluate_system(fed_system: FedGATSageSystem, args) -> dict:
    """Evaluate the trained federated system"""
    logger.info("Evaluating trained federated system")
    
    try:
        # For simplicity, we'll evaluate on one detector type's test data
        # In practice, you'd want ensemble evaluation
        primary_detector = args.detector_types[0]
        test_loader = fed_system.data_loaders[primary_detector]
        
        # Try to load test data
        test_data_path = os.path.join(args.data_dir, f'{primary_detector}_detector', 'test.csv')
        if not os.path.exists(test_data_path):
            logger.warning("No test data found for evaluation")
            return {}
        
        # Load and process test data
        test_data = test_loader._process_to_graph(pd.read_csv(test_data_path))
        
        if test_data is None or len(test_data['edge_labels']) == 0:
            logger.warning("Test data could not be processed")
            return {}
        
        # Get predictions from primary detector
        primary_model = fed_system.client_models[primary_detector][0]  # Use first client's model
        primary_model.eval()
        
        with torch.no_grad():
            x = test_data['features'].to(fed_system.device)
            edge_index = test_data['edge_index'].to(fed_system.device)
            edge_labels = test_data['edge_labels'].to(fed_system.device)
            
            _, edge_predictions = primary_model(x, edge_index)
            predicted_labels = edge_predictions.argmax(dim=1)
            
            # Calculate metrics
            y_true = edge_labels.cpu().numpy()
            y_pred = predicted_labels.cpu().numpy()
            
            # Get class names if available
            class_names = None
            if test_loader.label_mapper:
                class_names = [k for k, v in sorted(test_loader.label_mapper.items(), key=lambda x: x[1])]
            
            metrics = calculate_metrics(y_true, y_pred, class_names)
            
            # Create confusion matrix plot
            cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
            plot_confusion_matrix(y_true, y_pred, class_names, cm_path)
            
            logger.info(f"Evaluation complete - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['macro_f1']:.4f}")
            
            return metrics
            
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return {}

def create_visualizations(results: dict, output_dir: str):
    """Create visualization plots for the experiment results"""
    logger.info("Creating visualization plots")
    
    try:
        training_results = results.get('training', {})
        
        if 'training_losses' in training_results and 'round_times' in training_results:
            # Plot training progress
            progress_path = os.path.join(output_dir, 'training_progress.png')
            plot_training_progress(
                training_results['training_losses'],
                training_results['round_times'],
                progress_path
            )
        
        # Create performance summary plot
        evaluation_results = results.get('evaluation', {})
        if evaluation_results and 'per_class_detailed' in evaluation_results:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            classes = list(evaluation_results['per_class_detailed'].keys())
            f1_scores = [evaluation_results['per_class_detailed'][c]['f1'] for c in classes]
            
            bars = ax.bar(classes, f1_scores, alpha=0.7)
            ax.set_ylabel('F1 Score')
            ax.set_title('Per-Class F1 Scores')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, score in zip(bars, f1_scores):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{score:.3f}', ha='center', va='bottom')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            f1_path = os.path.join(output_dir, 'per_class_f1.png')
            plt.savefig(f1_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Per-class F1 plot saved to {f1_path}")
        
    except Exception as e:
        logger.error(f"Visualization creation failed: {e}")

def main():
    """Main experiment function"""
    args = parse_args()
    
    # Setup experiment environment
    device = setup_experiment(args)
    
    logger.info(f"Starting FedGATSage experiment with {args.dataset} dataset")
    
    # Demonstrate community abstraction concept
    demonstrate_community_abstraction(args.data_dir)
    
    # Run federated learning experiment
    results = run_federated_experiment(args, device)
    
    # Create visualizations
    create_visualizations(results, args.output_dir)
    
    # Save final results
    results_path = os.path.join(args.output_dir, 'final_results.json')
    save_results(results, results_path)
    
    # Print summary
    print("\n" + "="*60)
    print("FEDGATSAGE EXPERIMENT COMPLETED")
    print("="*60)
    
    if results.get('evaluation'):
        eval_results = results['evaluation']
        print(f"Final Accuracy: {eval_results.get('accuracy', 0):.4f}")
        print(f"Final F1 Score: {eval_results.get('macro_f1', 0):.4f}")
        print(f"Balanced Accuracy: {eval_results.get('balanced_accuracy', 0):.4f}")
    
    config = results.get('configuration', {})
    print(f"Configuration:")
    print(f"  - Clients: {config.get('num_clients', 'N/A')}")
    print(f"  - Rounds: {config.get('num_rounds', 'N/A')}")
    print(f"  - Detectors: {config.get('detector_types', 'N/A')}")
    
    print(f"\nResults saved to: {args.output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()