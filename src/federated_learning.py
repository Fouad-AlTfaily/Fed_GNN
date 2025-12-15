"""
Main federated learning orchestration for FedGATSage.
Handles client-server coordination, model aggregation, and flow embedding processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import time
import logging
import os

from gnn_models import TemporalGATDetector, ContentGATDetector, BehavioralGATDetector, GlobalGraphSAGE
from feature_engineering import FeatureEngineer, CentralityFeatureExtractor
from community_detection import CommunityAwareProcessor

logger = logging.getLogger(__name__)

class FlowEmbeddingGenerator:
    """Generates flow embeddings as community abstractions"""
    
    def __init__(self, detector_type: str = 'temporal'):
        self.detector_type = detector_type
        
    def generate_embeddings(self, model, data: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate flow embeddings from GAT node embeddings.
        This implements the community abstraction mechanism from Algorithm 1.
        """
        model.eval()
        with torch.no_grad():
            # Extract graph data
            x = data['features']
            edge_index = data['edge_index'] 
            edge_labels = data['edge_labels']
            
            logger.info(f"Generating embeddings for {x.shape[0]} nodes, {edge_index.shape[1]} edges")
            
            # Generate node embeddings using GAT
            try:
                node_embeddings, _ = model(x, edge_index)
            except Exception as e:
                logger.error(f"Error in GAT forward pass: {e}")
                return torch.empty(0), torch.empty(0)
            
            # Create flow embeddings (community abstractions)
            flow_embeddings = []
            flow_labels = []
            
            # Sample flows for efficiency and privacy
            unique_labels = torch.unique(edge_labels)
            max_per_class = min(250, len(edge_labels) // len(unique_labels))
            
            for label in unique_labels:
                mask = edge_labels == label
                if mask.sum() > 0:
                    label_indices = mask.nonzero(as_tuple=True)[0]
                    
                    # Sample representative flows
                    if len(label_indices) > max_per_class:
                        perm = torch.randperm(len(label_indices))[:max_per_class]
                        selected_indices = label_indices[perm]
                    else:
                        selected_indices = label_indices
                    
                    # Create flow embeddings for selected flows
                    for idx in selected_indices:
                        src_idx = edge_index[0, idx]
                        dst_idx = edge_index[1, idx]
                        
                        src_emb = node_embeddings[src_idx]
                        dst_emb = node_embeddings[dst_idx]
                        
                        # Flow embedding = community relationship abstraction
                        flow_emb = self._create_flow_embedding(src_emb, dst_emb, data, idx)
                        
                        flow_embeddings.append(flow_emb.unsqueeze(0))
                        flow_labels.append(label)
            
            if flow_embeddings:
                flow_embeddings = torch.cat(flow_embeddings, dim=0)
                flow_labels = torch.stack(flow_labels)
                
                logger.info(f"Generated {len(flow_embeddings)} flow embeddings")
                return flow_embeddings, flow_labels
            else:
                logger.warning("No flow embeddings generated")
                return torch.empty(0), torch.empty(0)
    
    def _create_flow_embedding(self, src_emb: torch.Tensor, dst_emb: torch.Tensor, 
                              data: Dict[str, Any], idx: int) -> torch.Tensor:
        """
        Create flow embedding representing community relationship.
        Implements Step 4 of Algorithm 1.
        """
        # Base embedding: concatenate source and destination
        embedding_parts = [src_emb, dst_emb]
        
        # Add interaction features (community relationship indicators)
        embedding_parts.append(src_emb * dst_emb)  # Element-wise product
        embedding_parts.append(torch.abs(src_emb - dst_emb))  # Absolute difference
        
        # Add traffic features if available
        if 'traffic_features' in data and data['traffic_features'] is not None:
            traffic_features = data['traffic_features'][idx]
            embedding_parts.append(traffic_features)
        
        # Combine all parts into flow embedding
        return torch.cat(embedding_parts)

class DataLoader:
    """Load and process data for FedGATSage clients"""
    
    def __init__(self, data_dir: str, detector_type: str = 'temporal'):
        self.data_dir = data_dir
        self.detector_type = detector_type
        self.feature_engineer = FeatureEngineer(detector_type)
        self.centrality_extractor = CentralityFeatureExtractor()
        self.community_processor = CommunityAwareProcessor()
        self.label_mapper = None
    
    def load_client_data(self, client_id: int) -> Optional[Dict[str, Any]]:
        """Load and process client data"""
        client_path = os.path.join(self.data_dir, f'client_{client_id}.csv')
        
        if not os.path.exists(client_path):
            logger.error(f"Client file not found: {client_path}")
            return None
        
        try:
            # Load raw data
            df = pd.read_csv(client_path)
            logger.info(f"Loaded {len(df)} records for client {client_id}")
            
            # Create label mapper if needed
            if self.label_mapper is None:
                self._create_label_mapper(df)
            
            # Apply feature engineering
            df = self.feature_engineer.extract_features(df)
            df = self.centrality_extractor.extract_centrality_features(df)
            
            # Add community-aware features (bridge to paper's Algorithm 1)
            df = self.community_processor.create_community_enhanced_features(df, {})
            
            # Convert to graph format
            return self._process_to_graph(df)
            
        except Exception as e:
            logger.error(f"Error loading client {client_id} data: {e}")
            return None
    
    def _create_label_mapper(self, df: pd.DataFrame):
        """Create consistent label mapping across clients"""
        unique_attacks = sorted(df['Attack'].unique())
        self.label_mapper = {attack: idx for idx, attack in enumerate(unique_attacks)}
        logger.info(f"Created label mapper with {len(self.label_mapper)} classes")
    
    def _process_to_graph(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Convert DataFrame to graph format for GNN processing"""
        # Get unique IPs as nodes
        unique_ips = pd.concat([df['Src IP'], df['Dst IP']]).unique()
        ip_to_idx = {ip: idx for idx, ip in enumerate(unique_ips)}
        
        # Extract node features (community-aware centrality measures)
        feature_cols = [col for col in df.columns if any(measure in col.lower() for measure in [
            'betweenness', 'pagerank', 'degree', 'closeness', 'eigenvector',
            'k_core', 'k_truss', 'modularity', 'flow_rate', 'avg_payload'
        ])]
        
        if not feature_cols:
            # Fallback to basic features
            feature_cols = ['flow_rate', 'avg_payload_fwd', 'protocol_encoded']
            for col in feature_cols:
                if col not in df.columns:
                    df[col] = 0.0
        
        # Create node features by averaging over IP addresses
        features = []
        for ip in unique_ips:
            ip_rows = df[(df['Src IP'] == ip) | (df['Dst IP'] == ip)]
            avg_features = ip_rows[feature_cols].mean().fillna(0.0).values
            features.append(avg_features)
        
        features = torch.tensor(np.array(features), dtype=torch.float32)
        
        # Create edges from flows
        edges = []
        edge_labels = []
        
        for _, row in df.iterrows():
            src_ip, dst_ip = row['Src IP'], row['Dst IP']
            if src_ip in ip_to_idx and dst_ip in ip_to_idx:
                src_idx = ip_to_idx[src_ip]
                dst_idx = ip_to_idx[dst_ip]
                edges.append([src_idx, dst_idx])
                edge_labels.append(self.label_mapper[row['Attack']])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_labels = torch.tensor(edge_labels, dtype=torch.long)
        
        return {
            'features': features,
            'edge_index': edge_index,
            'edge_labels': edge_labels,
            'ip_to_idx': ip_to_idx,
            'df': df
        }

class FedGATSageSystem:
    """Main FedGATSage federated learning system"""
    
    def __init__(self, data_dir: str, num_clients: int = 5, 
                 detector_types: List[str] = ['temporal', 'content', 'behavioral'],
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.data_dir = data_dir
        self.num_clients = num_clients
        self.detector_types = detector_types
        self.device = device
        
        # Initialize components for each detector type
        self.client_models = {}
        self.data_loaders = {}
        self.flow_generators = {}
        
        for detector_type in detector_types:
            detector_dir = os.path.join(data_dir, f'{detector_type}_detector')
            self.data_loaders[detector_type] = DataLoader(detector_dir, detector_type)
            self.flow_generators[detector_type] = FlowEmbeddingGenerator(detector_type)
            self.client_models[detector_type] = {}
        
        self.global_model = None
        self.results = {'training_losses': [], 'round_times': []}
        
        logger.info(f"Initialized FedGATSage with {len(detector_types)} detector types")
    
    def initialize_models(self, input_dim: int = 64, hidden_dim: int = 256, num_classes: int = 8):
        """Initialize client and server models"""
        
        for detector_type in self.detector_types:
            self.client_models[detector_type] = {}
            
            for client_id in range(self.num_clients):
                # Create specialized GAT model based on detector type
                if detector_type == 'temporal':
                    model = TemporalGATDetector(input_dim, hidden_dim, num_classes=num_classes)
                elif detector_type == 'content':
                    model = ContentGATDetector(input_dim, hidden_dim, num_classes=num_classes)
                elif detector_type == 'behavioral':
                    model = BehavioralGATDetector(input_dim, hidden_dim, num_classes=num_classes)
                
                self.client_models[detector_type][client_id] = model.to(self.device)
        
        # Determine flow embedding dimension for global GraphSAGE
        sample_client_data = self.data_loaders[self.detector_types[0]].load_client_data(1)
        if sample_client_data:
            sample_model = self.client_models[self.detector_types[0]][0]
            flow_gen = self.flow_generators[self.detector_types[0]]
            
            with torch.no_grad():
                sample_embeddings, _ = flow_gen.generate_embeddings(sample_model, sample_client_data)
                if len(sample_embeddings) > 0:
                    flow_embedding_dim = sample_embeddings.shape[1]
                else:
                    flow_embedding_dim = hidden_dim * 4  # Fallback
        else:
            flow_embedding_dim = hidden_dim * 4
        
        # Initialize global GraphSAGE model
        self.global_model = GlobalGraphSAGE(
            input_dim=flow_embedding_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes
        ).to(self.device)
        
        logger.info(f"Initialized models with flow embedding dim: {flow_embedding_dim}")
    
    def train_federated(self, num_rounds: int = 20) -> Dict[str, Any]:
        """Main federated training loop"""
        logger.info(f"Starting federated training for {num_rounds} rounds")
        
        for round_idx in range(num_rounds):
            round_start = time.time()
            logger.info(f"Starting round {round_idx + 1}/{num_rounds}")
            
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
            
            logger.info(f"Round {round_idx + 1} completed in {round_time:.2f}s, loss: {global_loss:.4f}")
        
        logger.info("Federated training completed")
        return self.results
    
    def _collect_client_updates(self, detector_type: str) -> List[Dict[str, Any]]:
        """Collect updates from clients for specific detector type"""
        client_updates = []
        
        for client_id in range(self.num_clients):
            # Load client data
            client_data = self.data_loaders[detector_type].load_client_data(client_id + 1)
            if client_data is None:
                continue
            
            client_model = self.client_models[detector_type][client_id]
            
            # Train client model locally
            metrics = self._train_client_model(client_model, client_data)
            
            # Generate flow embeddings (community abstractions)
            flow_gen = self.flow_generators[detector_type]
            flow_embeddings, flow_labels = flow_gen.generate_embeddings(client_model, client_data)
            
            if len(flow_embeddings) > 0:
                client_updates.append({
                    'client_id': client_id,
                    'detector_type': detector_type,
                    'flow_embeddings': flow_embeddings,
                    'flow_labels': flow_labels,
                    'model_state': client_model.state_dict(),
                    'metrics': metrics
                })
        
        return client_updates
    
    def _train_client_model(self, model, data) -> Dict[str, float]:
        """Train a single client model"""
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        x = data['features'].to(self.device)
        edge_index = data['edge_index'].to(self.device)
        edge_labels = data['edge_labels'].to(self.device)
        
        # Simple training loop for a few epochs
        for _ in range(5):
            optimizer.zero_grad()
            _, predictions = model(x, edge_index)
            loss = criterion(predictions, edge_labels)
            loss.backward()
            optimizer.step()
            
        return {'loss': loss.item()}
    
    def _aggregate_updates(self, client_updates: List[Dict[str, Any]]) -> float:
        """Aggregate updates using global GraphSAGE model"""
        if not client_updates:
            return 0.0
            
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
        
        # Create a fully connected graph for the global model (simplified)
        # In a real scenario, we would use the community structure to define edges
        num_nodes = global_x.shape[0]
        edge_index = torch.combinations(torch.arange(num_nodes), r=2).t().to(self.device)
        
        # Train global model
        self.global_model.train()
        optimizer = torch.optim.Adam(self.global_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        optimizer.zero_grad()
        _, predictions = self.global_model(global_x, edge_index)
        loss = criterion(predictions, global_y)
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def _redistribute_models(self):
        """Redistribute global knowledge back to clients"""
        # In this architecture, the global model learns to classify flows based on embeddings.
        # We can redistribute the knowledge by averaging the client models, 
        # weighted by their contribution to the global model performance.
        
        # For simplicity in this reference implementation, we use simple averaging
        # of the client models for each detector type.
        
        for detector_type in self.detector_types:
            client_states = []
            for client_id in self.client_models[detector_type]:
                client_states.append(self.client_models[detector_type][client_id].state_dict())
            
            if not client_states:
                continue
                
            # Simple averaging (can be enhanced with performance weighting)
            averaged_state = {}
            for key in client_states[0].keys():
                # Stack all client tensors for this key
                stacked = torch.stack([state[key] for state in client_states])
                
                # Handle non-floating point tensors (e.g. LongTensor for buffers)
                if not stacked.is_floating_point():
                    # Cast to float for averaging, then back to original type
                    averaged_state[key] = stacked.float().mean(0).type(stacked.dtype)
                else:
                    averaged_state[key] = stacked.mean(0)
            
            # Update all clients with averaged state
            for client_id in self.client_models[detector_type]:
                self.client_models[detector_type][client_id].load_state_dict(averaged_state)