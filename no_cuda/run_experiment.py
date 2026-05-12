"""
Wrapper script to run the FedGATSage experiment with CPU and compatibility fixes.

This script monkey-patches the following issues found when running with modern 
library versions (numpy>=2.0, networkx>=3.0, torch>=2.0):

1. networkx 3.x `modularity()` API change: requires partition (list of sets) 
   instead of community labels list
2. numpy 2.x `np.array_split` returns arrays instead of DataFrames
3. Evaluation skipping feature engineering (dimension mismatch)

Usage (from no_cuda/ directory):
    python run_experiment.py --demo_mode --data_dir ../data --device cpu [other args...]
    
All original source files remain UNCHANGED.
"""

import sys
import os
import warnings
import torch
import pandas as pd
import numpy as np

# Determine paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)  # parent = Fed_GNN root

# Suppress torch.testing deprecation warnings (harmless in this codebase)
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')
warnings.filterwarnings('ignore', category=UserWarning, module='torch')

# ============================================================
# FIX 1: NetworkX 3.x modularity API compatibility
# The original code passes `communities.values()` (list of int labels)
# to nx.community.modularity(), but NetworkX 3.x expects a list of sets.
# ============================================================
import networkx as nx
from typing import Dict

_original_compute_modularity_vitality = None

def _patch_community_detection():
    """
    Monkey-patch CommunityAwareProcessor.compute_modularity_vitality 
    to work with NetworkX 3.x API.
    """
    from community_detection import CommunityAwareProcessor
    
    global _original_compute_modularity_vitality
    
    if _original_compute_modularity_vitality is not None:
        return  # Already patched
    
    _original_compute_modularity_vitality = CommunityAwareProcessor.compute_modularity_vitality
    
    def patched_compute_modularity_vitality(self, graph: nx.Graph, 
                                           communities: Dict) -> Dict:
        """
        Fixed version that converts community labels to a proper partition
        (list of node sets) for NetworkX 3.x compatibility.
        """
        modularity_vitality = {}
        
        # Fix: Convert communities dict (node -> community_label) 
        # into partition format (list of sets) required by NetworkX 3.x
        comm_to_nodes = {}
        for node, comm_id in communities.items():
            comm_to_nodes.setdefault(comm_id, set()).add(node)
        partition = list(comm_to_nodes.values())
        
        base_modularity = nx.community.modularity(graph, partition)
        
        for node in graph.nodes():
            temp_graph = graph.copy()
            temp_graph.remove_node(node)
            temp_communities = {k: v for k, v in communities.items() if k != node}
            
            if len(temp_communities) > 0:
                # Also fix the same issue for the temp graph
                temp_comm_to_nodes = {}
                for n, comm_id in temp_communities.items():
                    temp_comm_to_nodes.setdefault(comm_id, set()).add(n)
                temp_partition = list(temp_comm_to_nodes.values())
                new_modularity = nx.community.modularity(temp_graph, temp_partition)
                modularity_vitality[node] = base_modularity - new_modularity
            else:
                modularity_vitality[node] = 0.0
        
        self.modularity_vitality = modularity_vitality
        return modularity_vitality
    
    CommunityAwareProcessor.compute_modularity_vitality = patched_compute_modularity_vitality
    print("[PATCH] Applied NetworkX 3.x modularity fix")


# ============================================================
# Now run the original experiment
# ============================================================
if __name__ == '__main__':
    # Add src to path (from root directory)
    sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))
    
    # Apply the community detection patch before importing anything that uses it
    _patch_community_detection()
    
    # Monkey-patch numpy array_split to preserve DataFrames (numpy 2.x fix)
    _orig_array_split = np.array_split
    def _patched_array_split(ary, indices_or_sections, axis=0):
        result = _orig_array_split(ary, indices_or_sections, axis=axis)
        if isinstance(ary, pd.DataFrame):
            return [pd.DataFrame(r, columns=ary.columns) for r in result]
        return result
    np.array_split = _patched_array_split
    
    # Rewrite sys.argv so argparse in the original experiment works correctly
    # (replace this script's name with the original experiment's name)
    sys.argv = [
        sys.argv[0].replace('run_experiment.py', '../experiments/fedgatsage_experiment.py')
    ] + sys.argv[1:]
    
    # Load and run the original experiment module
    import importlib.util
    experiment_path = os.path.join(ROOT_DIR, 'experiments', 'fedgatsage_experiment.py')
    spec = importlib.util.spec_from_file_location("fedgatsage_experiment", experiment_path)
    experiment_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment_module)
    
    # ============================================================
    # FIX 2: Patch evaluate_system to apply feature engineering to test data
    # The original code calls _process_to_graph directly on raw test CSV,
    # skipping the feature engineering pipeline. This causes dimension mismatch.
    # ============================================================
    _original_evaluate_system = experiment_module.evaluate_system
    
    def patched_evaluate_system(fed_system, args) -> dict:
        """Patched evaluation that applies feature engineering to test data."""
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Evaluating trained federated system")
        
        try:
            primary_detector = args.detector_types[0]
            test_loader = fed_system.data_loaders[primary_detector]
            
            test_data_path = os.path.join(args.data_dir, f'{primary_detector}_detector', 'test.csv')
            if not os.path.exists(test_data_path):
                logger.warning("No test data found for evaluation")
                return {}
            
            df_test = pd.read_csv(test_data_path)
            if args.demo_mode:
                df_test = df_test.head(1000)
            
            # FIX: Apply the same feature engineering pipeline as training
            df_test = test_loader.feature_engineer.extract_features(df_test)
            df_test = test_loader.centrality_extractor.extract_centrality_features(df_test)
            df_test = test_loader.community_processor.create_community_enhanced_features(df_test, {})
            
            test_data = test_loader._process_to_graph(df_test)
            
            if test_data is None or len(test_data['edge_labels']) == 0:
                logger.warning("Test data could not be processed")
                return {}
            
            primary_model = fed_system.client_models[primary_detector][0]
            primary_model.eval()
            
            with torch.no_grad():
                x = test_data['features'].to(fed_system.device)
                edge_index = test_data['edge_index'].to(fed_system.device)
                edge_labels = test_data['edge_labels'].to(fed_system.device)
                
                _, edge_predictions = primary_model(x, edge_index)
                predicted_labels = edge_predictions.argmax(dim=1)
                
                y_true = edge_labels.cpu().numpy()
                y_pred = predicted_labels.cpu().numpy()
                
                class_names = None
                if test_loader.label_mapper:
                    class_names = [k for k, v in sorted(test_loader.label_mapper.items(), key=lambda x: x[1])]
                
                from utils import calculate_metrics, plot_confusion_matrix
                metrics = calculate_metrics(y_true, y_pred, class_names)
                
                cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
                plot_confusion_matrix(y_true, y_pred, class_names, cm_path)
                
                logger.info(f"Evaluation complete - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['macro_f1']:.4f}")
                return metrics
                
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    experiment_module.evaluate_system = patched_evaluate_system
    
    # The original experiment uses `if __name__ == '__main__':` which won't trigger
    # when loaded as a module, so we replicate the main block here:
    args = experiment_module.parse_args()
    device = experiment_module.setup_experiment(args)
    experiment_module.demonstrate_community_abstraction(args.data_dir)
    results = experiment_module.run_federated_experiment(args, device)
    experiment_module.create_visualizations(results, args.output_dir)
    print("Experiment completed successfully!")
