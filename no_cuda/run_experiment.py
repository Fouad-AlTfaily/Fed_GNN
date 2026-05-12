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
import time
import atexit
import logging
import torch
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

# Determine paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)  # parent = Fed_GNN root

# Suppress torch.testing deprecation warnings (harmless in this codebase)
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')
warnings.filterwarnings('ignore', category=UserWarning, module='torch')

# ============================================================
# NTFY.SH NOTIFICATION SYSTEM
# Sends live experiment logs to ntfy.sh/asfi-fed-gnn
# ============================================================
NTFY_TOPIC = "asfi-fed-gnn"
NTFY_URL = f"https://ntfy.sh/{NTFY_TOPIC}"
_ntfy_session = None
_ntfy_available = False

def _init_ntfy():
    """Initialize ntfy session and test connectivity."""
    global _ntfy_session, _ntfy_available
    try:
        _ntfy_session = requests.Session()
        resp = _ntfy_session.get(NTFY_URL, timeout=5)
        _ntfy_available = resp.status_code == 200
        if _ntfy_available:
            print(f"[NTFY] Connected to {NTFY_URL}", flush=True)
            ntfy_send("🚀 FedGATSage experiment started", 
                      title="Experiment Start", priority=3, tags="rocket")
    except Exception:
        _ntfy_available = False
        print("[NTFY] Notifications unavailable (ntfy.sh unreachable)", flush=True)

def ntfy_send(message, title="FedGATSage", priority=3, tags="robot"):
    """Send a notification to ntfy.sh/asfi-fed-gnn. Fails silently if unreachable."""
    global _ntfy_session, _ntfy_available
    if not _ntfy_available:
        return
    try:
        _ntfy_session.post(
            NTFY_URL,
            data=message.encode('utf-8'),
            headers={
                "Title": title[:250],
                "Priority": str(min(max(priority, 1), 5)),
                "Tags": tags
            },
            timeout=10
        )
    except Exception:
        pass

class NtfyLogHandler(logging.Handler):
    """Custom logging handler that forwards all log messages to ntfy.sh."""
    def emit(self, record):
        try:
            msg = self.format(record)
            # Send all levels; use emoji for severity
            emoji = {"DEBUG": "🔍", "INFO": "ℹ️", "WARNING": "⚠️", "ERROR": "❌", "CRITICAL": "🔥"}.get(record.levelname, "📝")
            prio = 5 if record.levelno >= logging.CRITICAL else (4 if record.levelno >= logging.ERROR else (3 if record.levelno >= logging.WARNING else 2))
            ntfy_send(f"{emoji} {msg}", 
                      title=f"FedGATSage {record.levelname}", 
                      priority=prio,
                      tags="warning" if record.levelno >= logging.WARNING else "information_source")
        except Exception:
            pass

# Ensure all stdout/stderr is line-buffered for interactive feel
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None

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
# FIX 3: Patch _aggregate_updates to subsample edges for the global GraphSAGE
# The original code creates a fully-connected graph with torch.combinations(),
# which produces O(n²) edges. On NF-ToN-IoT (~90K rows), this generates
# ~450M edges and ~3.6GB — causing a hang on CPU.
# ============================================================
MAX_GLOBAL_NODES = 2000  # Maximum nodes in global aggregation graph

_original_aggregate_updates = None

def _patch_aggregate_updates():
    """Monkey-patch FedGATSageSystem._aggregate_updates to subsample edges."""
    from federated_learning import FedGATSageSystem
    global _original_aggregate_updates
    
    if _original_aggregate_updates is not None:
        return
    
    _original_aggregate_updates = FedGATSageSystem._aggregate_updates
    
    def patched_aggregate_updates(self, client_updates):
        """Fixed version that subsamples nodes to avoid O(n²) blowup."""
        if not client_updates:
            return 0.0
        
        import logging
        log = logging.getLogger(__name__)
        
        # Prepare batch for global model
        all_embeddings = []
        all_labels = []
        
        for update in client_updates:
            all_embeddings.append(update['flow_embeddings'].to(self.device))
            all_labels.append(update['flow_labels'].to(self.device))
        
        if not all_embeddings:
            return 0.0
        
        # Concatenate all flow embeddings
        global_x = torch.cat(all_embeddings, dim=0)
        global_y = torch.cat(all_labels, dim=0)
        
        num_nodes = global_x.shape[0]
        log.info(f"Aggregating {len(client_updates)} client updates with {num_nodes} total flow embeddings")
        
        # FIX: Subsample nodes if too many (avoids O(n²) edge blowup)
        if num_nodes > MAX_GLOBAL_NODES:
            log.warning(f"Subsampling global nodes from {num_nodes} to {MAX_GLOBAL_NODES} "
                        f"(avoids O(n²) edge blowup in torch.combinations)")
            perm = torch.randperm(num_nodes)[:MAX_GLOBAL_NODES]
            global_x = global_x[perm]
            global_y = global_y[perm]
            num_nodes = global_x.shape[0]
            ntfy_send(f"⚡ Subsampled global graph: {num_nodes} nodes (from {len(perm)} total)",
                      title="FedGATSage MemSaver", priority=2, tags="zap")
        
        # Create a sparse graph for the global model (not fully-connected)
        # Use random edges with limited degree to keep things manageable
        import math
        max_edges = min(num_nodes * 50, 100000)  # Max 50 edges per node, 100K total
        log.info(f"Creating global graph with {num_nodes} nodes and ~{max_edges} edges")
        
        # Generate random edges (sparse graph instead of fully-connected)
        edge_src = torch.randint(0, num_nodes, (max_edges,), device=self.device)
        edge_dst = torch.randint(0, num_nodes, (max_edges,), device=self.device)
        # Remove self-loops
        mask = edge_src != edge_dst
        edge_src = edge_src[mask]
        edge_dst = edge_dst[mask]
        edge_index = torch.stack([edge_src, edge_dst], dim=0)
        
        log.info(f"Global edge_index shape: {edge_index.shape} (was O(n²) before patch)")
        
        # Train global model
        self.global_model.train()
        optimizer = torch.optim.Adam(self.global_model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        optimizer.zero_grad()
        _, predictions = self.global_model(global_x, edge_index)
        loss = criterion(predictions, global_y)
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    FedGATSageSystem._aggregate_updates = patched_aggregate_updates
    print("[PATCH] Applied global GraphSAGE edge subsampling fix (avoids O(n²) hang)", flush=True)


# ============================================================
# FIX 4: Patch train_federated to show tqdm progress + send ntfy round updates
# ============================================================
_original_train_federated = None

def _patch_train_federated():
    """Monkey-patch FedGATSageSystem.train_federated to add tqdm and ntfy updates."""
    from federated_learning import FedGATSageSystem
    global _original_train_federated
    
    if _original_train_federated is not None:
        return
    
    _original_train_federated = FedGATSageSystem.train_federated
    
    def patched_train_federated(self, num_rounds: int = 20):
        """Patched version with tqdm progress bar and ntfy round updates."""
        import logging
        log = logging.getLogger(__name__)
        log.info(f"Starting federated training for {num_rounds} rounds")
        ntfy_send(f"🏋️ Training started: {num_rounds} rounds", 
                  title="FedGATSage Training", priority=2, tags="hourglass_flowing_sand")
        
        pbar = tqdm(range(num_rounds), desc="Federated Training", unit="round",
                    ncols=100, file=sys.stdout, dynamic_ncols=False)
        
        for round_idx in pbar:
            round_start = time.time()
            log.info(f"Starting round {round_idx + 1}/{num_rounds}")
            
            # Collect updates from all clients across all detector types
            all_client_updates = []
            
            for detector_type in self.detector_types:
                client_updates = self._collect_client_updates(detector_type)
                all_client_updates.extend(client_updates)
            
            # Server-side aggregation with GraphSAGE
            global_loss = self._aggregate_updates(all_client_updates)
            
            # Redistribute updated parameters
            self._redistribute_models()
            
            round_time = time.time() - round_start
            self.results['training_losses'].append(global_loss)
            self.results['round_times'].append(round_time)
            
            # Update tqdm description with metrics
            pbar.set_postfix({
                'loss': f'{global_loss:.4f}',
                'time': f'{round_time:.1f}s',
                'updates': len(all_client_updates)
            })
            
            log.info(f"Round {round_idx + 1} completed in {round_time:.2f}s, loss: {global_loss:.4f}")
            
            # Send ntfy update for every round
            ntfy_send(f"📊 Round {round_idx + 1}/{num_rounds} | Loss: {global_loss:.4f} | ⏱️ {round_time:.1f}s | Updates: {len(all_client_updates)}",
                      title=f"FedGATSage Round {round_idx + 1}", priority=2, tags="bar_chart")
        
        pbar.close()
        log.info("Federated training completed")
        ntfy_send(f"✅ Training complete | Final loss: {self.results['training_losses'][-1]:.4f} | "
                  f"Total: {sum(self.results['round_times']):.0f}s",
                  title="FedGATSage Training Done", priority=4, tags="white_check_mark")
        return self.results
    
    FedGATSageSystem.train_federated = patched_train_federated
    print("[PATCH] Applied tqdm progress + ntfy round updates to training loop", flush=True)


# ============================================================
# FIX 5: Patch create_community_enhanced_features to avoid O(n²) graph
# building and modularity vitality computation that hangs on real datasets.
# - Replaces iterrows() with faster edge-list building
# - Skips modularity vitality when graph has > 500 nodes (too expensive)
# - Adds progress logging so user knows it's not stuck
# ============================================================
_original_create_community_enhanced = None

def _patch_community_features():
    """Monkey-patch CommunityAwareProcessor.create_community_enhanced_features for speed."""
    from community_detection import CommunityAwareProcessor
    global _original_create_community_enhanced
    
    if _original_create_community_enhanced is not None:
        return
    
    _original_create_community_enhanced = CommunityAwareProcessor.create_community_enhanced_features
    
    def patched_create_community_enhanced_features(self, df, ip_to_idx=None):
        """Faster version that avoids iterrows() and skips expensive modularity vitality for large graphs."""
        import logging
        log = logging.getLogger(__name__)
        
        n_rows = len(df)
        log.info(f"Building community graph from {n_rows} flows...")
        print(f"  Building community graph from {n_rows:,} flows...", flush=True)
        
        # FIX: Use fast edge-list building instead of iterrows()
        # Create edge list with counts using pandas groupby
        edges_df = df.groupby(['Src IP', 'Dst IP']).size().reset_index(name='weight')
        n_unique_edges = len(edges_df)
        log.info(f"  {n_unique_edges:,} unique edges found")
        print(f"  {n_unique_edges:,} unique edges found", flush=True)
        
        # Build graph using add_edges_from (much faster than iterrows)
        G = nx.Graph()
        G.add_edges_from(
            (row['Src IP'], row['Dst IP'], {'weight': row['weight']})
            for _, row in edges_df.iterrows()
        )
        n_nodes = G.number_of_nodes()
        log.info(f"  Graph built: {n_nodes:,} nodes, {G.number_of_edges():,} edges")
        print(f"  Graph built: {n_nodes:,} nodes, {G.number_of_edges():,} edges", flush=True)
        
        # Step 2: Detect communities
        log.info("  Running Louvain community detection...")
        print("  Running Louvain community detection...", flush=True)
        communities = self.detect_communities_louvain(G)
        
        # Step 3: Compute modularity vitality (SKIP if too many nodes)
        mod_vitality = {}
        if n_nodes <= 500:
            log.info(f"  Computing modularity vitality for {n_nodes} nodes...")
            print(f"  Computing modularity vitality for {n_nodes} nodes...", flush=True)
            mod_vitality = self.compute_modularity_vitality(G, communities)
        else:
            log.warning(f"  Skipping modularity vitality for {n_nodes} nodes (too expensive, > 500)")
            print(f"  ⚠ Skipping modularity vitality ({n_nodes} nodes > 500 limit)", flush=True)
            ntfy_send(f"⚡ Skipped modularity vitality for {n_nodes} nodes (would be O(n²))",
                      title="FedGATSage Perf", priority=2, tags="zap")
        
        # Steps 4-5: Add community features
        df_enhanced = df.copy()
        df_enhanced['src_community'] = df_enhanced['Src IP'].map(communities)
        df_enhanced['dst_community'] = df_enhanced['Dst IP'].map(communities)
        
        if mod_vitality:
            df_enhanced['src_mod_vitality'] = df_enhanced['Src IP'].map(mod_vitality)
            df_enhanced['dst_mod_vitality'] = df_enhanced['Dst IP'].map(mod_vitality)
        else:
            df_enhanced['src_mod_vitality'] = 0.0
            df_enhanced['dst_mod_vitality'] = 0.0
        
        df_enhanced['is_inter_community'] = (
            df_enhanced['src_community'] != df_enhanced['dst_community']
        ).astype(int)
        
        log.info(f"Enhanced {len(df_enhanced)} flows with community features")
        print(f"  ✓ Community features added successfully", flush=True)
        return df_enhanced
    
    CommunityAwareProcessor.create_community_enhanced_features = patched_create_community_enhanced_features
    print("[PATCH] Applied fast community graph building + modularity vitality skip", flush=True)


# ============================================================
# Now run the original experiment
# ============================================================
if __name__ == '__main__':
    # Initialize ntfy notifications EARLY
    _init_ntfy()
    
    # Add src to path (from root directory)
    sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))
    
    # Apply the community detection patch before importing anything that uses it
    _patch_community_detection()
    
    # Apply the aggregate_updates patch (fixes the hang)
    _patch_aggregate_updates()
    
    # Apply the tqdm + ntfy progress patch for training loop
    _patch_train_federated()
    
    # Apply fast community features patch (avoids iterrows hang + modularity O(n²))
    _patch_community_features()
    
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
    
    # Load the original experiment module
    import importlib.util
    experiment_path = os.path.join(ROOT_DIR, 'experiments', 'fedgatsage_experiment.py')
    spec = importlib.util.spec_from_file_location("fedgatsage_experiment", experiment_path)
    experiment_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment_module)
    
    # Attach ntfy handler to root logger so ALL log messages go to ntfy
    root_logger = logging.getLogger()
    ntfy_handler = NtfyLogHandler()
    ntfy_handler.setLevel(logging.INFO)  # ALL levels go to ntfy
    ntfy_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
    root_logger.addHandler(ntfy_handler)
    
    # ============================================================
    # FIX 2 (original): Patch evaluate_system to apply feature engineering to test data
    # The original code calls _process_to_graph directly on raw test CSV,
    # skipping the feature engineering pipeline. This causes dimension mismatch.
    # ============================================================
    _original_evaluate_system = experiment_module.evaluate_system
    
    def patched_evaluate_system(fed_system, args) -> dict:
        """Patched evaluation that applies feature engineering to test data."""
        import logging
        log = logging.getLogger(__name__)
        log.info("Evaluating trained federated system")
        ntfy_send("📊 Starting system evaluation...", title="FedGATSage Eval", priority=2, tags="chart")
        
        try:
            primary_detector = args.detector_types[0]
            test_loader = fed_system.data_loaders[primary_detector]
            
            test_data_path = os.path.join(args.data_dir, f'{primary_detector}_detector', 'test.csv')
            if not os.path.exists(test_data_path):
                log.warning("No test data found for evaluation")
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
                log.warning("Test data could not be processed")
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
                
                log.info(f"Evaluation complete - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['macro_f1']:.4f}")
                ntfy_send(f"✅ Evaluation done | Acc: {metrics['accuracy']:.4f} | F1: {metrics['macro_f1']:.4f}",
                          title="FedGATSage Complete", priority=3, tags="white_check_mark")
                return metrics
                
        except Exception as e:
            log.error(f"Evaluation failed: {e}")
            ntfy_send(f"❌ Evaluation FAILED: {e}", title="FedGATSage Error", priority=5, tags="x")
            import traceback
            traceback.print_exc()
            return {}
    
    experiment_module.evaluate_system = patched_evaluate_system
    
    # The original experiment uses `if __name__ == '__main__':` which won't trigger
    # when loaded as a module, so we replicate the main block here:
    print("\n" + "="*60, flush=True)
    print("  FedGATSage Experiment Starting", flush=True)
    print("  Logs → ntfy.sh/asfi-fed-gnn", flush=True)
    print("="*60 + "\n", flush=True)
    
    args = experiment_module.parse_args()
    ntfy_send(f"📋 Config: {args.dataset}, {args.num_clients} clients, {args.num_rounds} rounds, device={args.device}",
              title="FedGATSage Config", priority=2, tags="gear")
    
    print(f"[1/5] Setting up experiment environment (device: {args.device})...", flush=True)
    device = experiment_module.setup_experiment(args)
    print(f"      ✓ Environment ready, using: {device}", flush=True)
    
    print(f"[2/5] Running community abstraction demo...", flush=True)
    experiment_module.demonstrate_community_abstraction(args.data_dir)
    print(f"      ✓ Community abstraction demo complete", flush=True)
    
    print(f"[3/5] Starting federated experiment (this may take a while on CPU)...", flush=True)
    print(f"      Building graphs, detecting communities, initializing models...", flush=True)
    results = experiment_module.run_federated_experiment(args, device)
    print(f"      ✓ Federated experiment complete", flush=True)
    
    print(f"[4/5] Creating visualizations...", flush=True)
    experiment_module.create_visualizations(results, args.output_dir)
    print(f"      ✓ Visualizations saved to {args.output_dir}", flush=True)
    
    print(f"[5/5] Finalizing...", flush=True)
    
    # Send final completion notification
    eval_results = results.get('evaluation', {})
    acc = eval_results.get('accuracy', 'N/A')
    f1 = eval_results.get('macro_f1', 'N/A')
    ntfy_send(f"🎉 Experiment FINISHED! | Acc: {acc} | F1: {f1} | Results: {args.output_dir}",
              title="FedGATSage Done ✅", priority=5, tags="tada,partying_face")

    # Cleanup
    atexit.register(lambda: ntfy_send("🛑 FedGATSage experiment process exiting",
                                       title="FedGATSage Exit", priority=2, tags="stop"))
