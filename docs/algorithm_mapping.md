# Algorithm Mapping: Paper to Code

This document provides a detailed mapping between the algorithms described in the FedGATSage paper and their implementation in the codebase.

## Overview

The FedGATSage paper presents two main algorithms:
- **Algorithm 1**: Community Detection and Flow Embedding Creation  
- **Algorithm 2**: Community Overlay Construction (server-side)

Our implementation achieves the same goals through flow-level community abstraction rather than explicit community detection steps.

## Algorithm 1: Community Detection and Flow Embedding Creation

### Paper Description vs Implementation

| Paper Step | Paper Description | Code Implementation | File Location |
|-------------|------------------|-------------------|---------------|
| Step 1 | `G = (V, E) ← ConstructGraph(D)` | Graph construction in `_process_to_graph()` | `src/federated_learning.py:DataLoader` |
| Step 2 | `Communities ← LouvainAlgorithm(G)` | `detect_communities_louvain()` | `src/community_detection.py` |
| Step 3 | `ModVitality[v] ← ComputeModularityVitality(v, Communities)` | `compute_modularity_vitality()` | `src/community_detection.py` |
| Step 4 | `H ← GAT(X, G)` | Specialized GAT variants | `src/gnn_models.py` |
| Step 5 | Flow embedding creation | `_create_flow_embedding()` | `src/federated_learning.py:FlowEmbeddingGenerator` |
| Step 6 | `E_c ← AggregateNodeEmbeddings(c, H)` | Implicit in flow sampling | `src/federated_learning.py` |

### Key Implementation Insight

Our **flow embeddings serve as community abstractions**:
- Each flow embedding represents a relationship between community-aware nodes
- Community structure is captured through centrality features (modularity, k-core)
- Privacy is achieved through flow sampling rather than explicit community aggregation

## Algorithm 2: Community Overlay Construction

### Paper Description vs Implementation

| Paper Step | Paper Description | Code Implementation | File Location |
|-------------|------------------|-------------------|---------------|
| Graph Init | `Initialize empty graph G_overlay` | `_build_overlay_graph()` | `src/federated_learning.py` |
| Node Addition | Add community nodes | Flow embeddings as nodes | `src/federated_learning.py` |
| Similarity | `S_{ij} = cosine(E_{c_i}, E_{c_j})` | Cosine similarity between flow embeddings | `src/federated_learning.py` |
| Edge Creation | Add edges based on similarity threshold | Similarity-based edge creation | `src/federated_learning.py` |
| GraphSAGE | Apply GraphSAGE to overlay | `GlobalGraphSAGE` processing | `src/gnn_models.py` |

## Specialized GAT Variants

### Paper Mention vs Implementation

The paper mentions three specialized GAT variants for different attack types:

| Detector Type | Target Attacks | Code Implementation |
|---------------|----------------|-------------------|
| Temporal GAT | DDoS, DoS, Scanning | `TemporalGATDetector` in `src/gnn_models.py` |
| Content GAT | Injection, XSS | `ContentGATDetector` in `src/gnn_models.py` |
| Behavioral GAT | Password, Backdoor, etc. | `BehavioralGATDetector` in `src/gnn_models.py` |

### Key Features:

- **Temporal GAT**: Specialized attention mechanism for time-based patterns
- **Content GAT**: Wider classifier architecture for content analysis  
- **Behavioral GAT**: Session pattern encoder for behavioral analysis

## Feature Engineering Correspondence

### Paper Features vs Code Implementation

| Paper Feature Type | Code Implementation | File Location |
|-------------------|-------------------|---------------|
| Community-aware centrality | `extract_centrality_features()` | `src/feature_engineering.py` |
| Temporal features | `_add_temporal_features()` | `src/feature_engineering.py` |
| Content features | `_add_content_features()` | `src/feature_engineering.py` |
| Behavioral features | `_add_behavioral_features()` | `src/feature_engineering.py` |

## Federated Learning Process

### Paper Workflow vs Implementation

| Paper Step | Code Implementation | File Location |
|------------|-------------------|---------------|
| Local GAT training | `_train_client_model()` | `src/federated_learning.py` |
| Flow embedding generation | `generate_embeddings()` | `src/federated_learning.py` |
| Server aggregation | `_aggregate_updates()` | `src/federated_learning.py` |
| GraphSAGE processing | `GlobalGraphSAGE.forward()` | `src/gnn_models.py` |
| Parameter redistribution | `_redistribute_models()` | `src/federated_learning.py` |

## Privacy Mechanisms

### Paper Claims vs Implementation

| Privacy Mechanism | Paper Description | Code Implementation |
|------------------|------------------|-------------------|
| Community abstraction | Share community-level representations | Flow embeddings with community-aware features |
| Individual device protection | No raw device data shared | IP addresses abstracted in flow embeddings |
| Structural pattern preservation | Maintain network relationships | Community-aware centrality measures |
| Communication efficiency | Reduced data transfer | Flow sampling in `generate_embeddings()` |

## Validation Points

To validate that our implementation matches the paper's intent:

1. **Community Structure**: Run `community_detection.py` to see explicit community detection
2. **Flow Abstraction**: Check `FlowEmbeddingGenerator` to see how flows represent communities
3. **Specialized Detection**: Test each GAT variant on its target attack types
4. **Privacy Preservation**: Verify no raw IP data leaves clients in federated process
5. **Performance**: Compare results with paper's reported metrics

## Running the Complete Pipeline

```bash
# Demonstrate community abstraction
python experiments/fedgatsage_experiment.py --data_dir /path/to/data --demo_mode

# Full experiment matching paper
python experiments/fedgatsage_experiment.py \
  --data_dir /path/to/data \
  --dataset cic_ton_iot \
  --num_clients 5 \
  --num_rounds 15 \
  --detector_types temporal content behavioral


  This mapping demonstrates that while our implementation uses flow-level abstraction rather than explicit community detection, it achieves the same privacy, performance, and architectural goals described in the FedGATSage paper.