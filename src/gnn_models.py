"""
Specialized GAT variants for FedGATSage: Temporal, Content, and Behavioral detectors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv
import logging

logger = logging.getLogger(__name__)

class TemporalGATDetector(nn.Module):
    """
    GAT specialized for temporal attacks (DDoS, DoS, Scanning).
    This model focuses on the timing and frequency of interactions.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, 
                 num_heads: int = 8, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        
        # We use multi-head attention to capture different types of temporal relationships
        # (e.g., one head for short bursts, another for periodic patterns)
        self.gat1 = GATConv(input_dim, hidden_dim // num_heads, 
                           heads=num_heads, concat=True, dropout=dropout)
        self.gat2 = GATConv(hidden_dim, hidden_dim, 
                           heads=1, concat=False, dropout=dropout)
        self.gat3 = GATConv(hidden_dim, hidden_dim, 
                           heads=1, concat=False, dropout=dropout)
        
        # Specialized attention mechanism for temporal weighting
        # This allows the model to focus on specific time windows or burst patterns
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Layer normalization helps with training stability, especially for deep GNNs
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        # The edge classifier determines if a specific flow (edge) represents an attack
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Residual connections prevent the "vanishing gradient" problem in deeper networks
        self.skip_conn1 = nn.Linear(input_dim, hidden_dim)
        self.skip_conn2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        # Residual connection 1
        res1 = self.skip_conn1(x)
        
        # First GAT layer with attention
        x = self.gat1(x, edge_index)
        x = self.norm1(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = x + res1
        
        # Apply temporal attention weighting
        temporal_weights = self.temporal_attention(x)
        x = x * temporal_weights
        
        # Second GAT layer
        res2 = self.skip_conn2(x)
        x = self.gat2(x, edge_index)
        x = self.norm2(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = x + res2
        
        # Third GAT layer
        x = self.gat3(x, edge_index)
        x = self.norm3(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Edge prediction for flow classification
        edge_src, edge_dst = edge_index
        edge_features = torch.cat([x[edge_src], x[edge_dst]], dim=1)
        edge_predictions = self.edge_classifier(edge_features)
        
        return x, edge_predictions

class ContentGATDetector(nn.Module):
    """GAT specialized for content attacks (Injection, XSS)"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, 
                 num_heads: int = 8, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        
        # Enhanced attention for content pattern detection
        self.gat1 = GATConv(input_dim, hidden_dim // num_heads, 
                           heads=num_heads, concat=True, dropout=dropout)
        self.gat2 = GATConv(hidden_dim, hidden_dim, 
                           heads=2, concat=False, dropout=dropout)
        self.gat3 = GATConv(hidden_dim, hidden_dim, 
                           heads=2, concat=False, dropout=dropout)
        
        # Content pattern attention
        self.content_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Wider classifier for content patterns
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 3),
            nn.BatchNorm1d(hidden_dim * 3),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, num_classes)
        )
        
        # Normalization and connections
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.skip_conn1 = nn.Linear(input_dim, hidden_dim)
        self.skip_conn2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        res1 = self.skip_conn1(x)
        
        x = self.gat1(x, edge_index)
        x = self.norm1(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = x + res1
        
        # Apply content-specific attention
        content_weights = self.content_attention(x)
        x = x * content_weights
        
        res2 = self.skip_conn2(x)
        x = self.gat2(x, edge_index)
        x = self.norm2(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = x + res2
        
        x = self.gat3(x, edge_index)
        x = self.norm3(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        edge_src, edge_dst = edge_index
        edge_features = torch.cat([x[edge_src], x[edge_dst]], dim=1)
        edge_predictions = self.edge_classifier(edge_features)
        
        return x, edge_predictions

class BehavioralGATDetector(nn.Module):
    """GAT specialized for behavioral attacks (Password, Backdoor, etc.)"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, 
                 num_heads: int = 8, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        
        self.gat1 = GATConv(input_dim, hidden_dim // num_heads, 
                           heads=num_heads, concat=True, dropout=dropout)
        self.gat2 = GATConv(hidden_dim, hidden_dim, 
                           heads=1, concat=False, dropout=dropout)
        self.gat3 = GATConv(hidden_dim, hidden_dim, 
                           heads=1, concat=False, dropout=dropout)
        
        # Session pattern encoder for behavioral analysis
        self.session_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.skip_conn1 = nn.Linear(input_dim, hidden_dim)
        self.skip_conn2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        res1 = self.skip_conn1(x)
        
        x = self.gat1(x, edge_index)
        x = self.norm1(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = x + res1
        
        # Encode session patterns for behavioral analysis
        session_features = self.session_encoder(x)
        x = x * session_features
        
        res2 = self.skip_conn2(x)
        x = self.gat2(x, edge_index)
        x = self.norm2(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = x + res2
        
        x = self.gat3(x, edge_index)
        x = self.norm3(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        edge_src, edge_dst = edge_index
        edge_features = torch.cat([x[edge_src], x[edge_dst]], dim=1)
        edge_predictions = self.edge_classifier(edge_features)
        
        return x, edge_predictions

class GlobalGraphSAGE(nn.Module):
    """Server-side GraphSAGE for processing community flow embeddings"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        
        # Input projection for flow embeddings
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        # GraphSAGE layers
        self.sage1 = SAGEConv(hidden_dim * 2, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim // 2)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        
        # Global classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x, edge_index):
        # Process flow embeddings
        x = self.input_projection(x)
        
        # GraphSAGE layers
        x = self.sage1(x, edge_index)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.dropout(x)
        
        x = self.sage2(x, edge_index)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.dropout(x)
        
        # Global classification
        embeddings = x
        predictions = self.classifier(x)
        
        return embeddings, predictions



