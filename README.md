# FedGATSage: Graph-based Federated Learning for IoT Intrusion Detection

## Overview
Implementation of FedGATSage, a federated learning architecture combining client-side GAT with server-side GraphSAGE through community abstraction for privacy-preserving IoT intrusion detection.

## Key Innovation: Community Abstraction via Flow Embeddings

Our approach achieves community-level privacy protection through flow-level abstraction:
- **Flow embeddings** represent relationships between community-aware nodes
- **Community features** (modularity, k-core) encode structural patterns
- **Sampling strategy** provides additional privacy layer
- **No raw device data** leaves client networks

## Quick Start

```bash
pip install -r requirements.txt
python experiments/fedgatsage_experiment.py --dataset cic_ton_iot