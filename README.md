# FedGATSage: Graph-based Federated Learning for IoT Intrusion Detection

[![Paper](https://img.shields.io/badge/Paper-Scientific%20Reports-red)](https://doi.org/10.1038/s41598-025-25175-1)
[![License](https://img.shields.io/badge/License-Open%20Access-green)](http://creativecommons.org/licenses/by-nc-nd/4.0/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8.0+-orange)](https://pytorch.org/)

Official implementation of **"Graph-based federated learning approach for intrusion detection in IoT networks"** published in *Scientific Reports* (2025).

**Authors:** Fouad Al Tfaily, Zakariya Ghalmane, Mohamed el Amine Brahmia, Hussein Hazimeh, Ali Jaber, Mourad Zghal

---

## 📋 Abstract

FedGATSage addresses critical limitations in federated learning for IoT intrusion detection where traditional approaches using LSTM/CNN cannot capture structural patterns, and GNN-based federated methods lose temporal patterns during parameter aggregation. Our hybrid architecture integrates client-side Graph Attention Networks (GAT) with server-side GraphSAGE through community abstraction, achieving:

- **78.58% balanced accuracy** on NF-ToN-IoT (80.24% on CIC-ToN-IoT)
- **85% communication overhead reduction**
- **Successful detection of coordinated attacks** (DDoS, DoS) that existing federated methods cannot handle
- **Privacy preservation** through community-based embeddings

---

## 🎯 Key Innovations

### 1. **Specialized GAT Detector Variants**
Three specialized architectures targeting different attack categories:

- **Temporal GAT**: Detects time-dependent attacks (DDoS, DoS) with perfect recall (1.0) for DoS
- **Content GAT**: Targets payload-based attacks (Injection, XSS) with near-perfect recall (0.999) for XSS  
- **Behavioral GAT**: Identifies behavioral attacks (Backdoor, Password, Scanning) with excellent recall (0.994) for Backdoor

### 2. **Community Abstraction Mechanism**
- Flow-level embeddings preserve structural and temporal patterns
- Community-level aggregation protects individual device identities
- Reduces data transfer from ~25KB to 3.2KB per client (85% reduction)

### 3. **Hybrid Federated Architecture**
- **Client-side GAT**: Captures local structural patterns and device relationships
- **Server-side GraphSAGE**: Learns global patterns across communities without destroying temporal signatures
- **Adaptive weighting**: Performance-based client contribution weighting addresses heterogeneity

### 4. **Enhanced Pattern Preservation**
- Maintains both structural relationships (network topology) and temporal sequences (attack evolution)
- Strong correlation (0.87) between community structure and attack patterns validates abstraction approach

---

## 📊 Performance Results

### Overall Performance

| Approach | NF-ToN-IoT ||| CIC-ToN-IoT |||
|----------|------------|------------|------|------------|------------|------|
| | Balanced Acc. | F1 | FPR/FNR | Balanced Acc. | F1 | FPR/FNR |
| **FedGATSage** | **0.785** | **0.619** | **0.066/0.223** | **0.802** | **0.800** | **0.039/0.197** |
| Centralized GAT | 0.811 | 0.797 | 0.026/0.188 | 0.840 | 0.820 | 0.024/0.169 |
| LSTM (Federated) | 0.713 | 0.610 | 0.052/0.286 | 0.759 | 0.752 | 0.043/0.225 |

**Key Achievement**: Only 2.8% performance gap compared to centralized models while maintaining complete privacy.

### Attack-Specific Detection (NF-ToN-IoT)

| Attack Type | Precision | Recall | F1 Score | FNR |
|-------------|-----------|--------|----------|-----|
| **Benign** | 0.9986 | 0.9944 | 0.9942 | 0.0056 |
| **Backdoor** | 0.9451 | 0.9942 | 0.9750 | 0.0058 |
| **DDoS** | 0.8939 | 0.5322 | 0.6354 | 0.4678 |
| **DoS** | 0.3896 | 1.0000 | 0.5721 | 0.0000 |
| **Injection** | 0.9760 | 0.3412 | 0.5100 | 0.6588 |
| **XSS** | 0.4197 | 0.9990 | 0.5900 | 0.0010 |

### Computational Efficiency

- **Training time**: 7.5 hours (vs. 5.5 hours centralized, 8.5 hours Fed-LSTM)
- **Memory**: 2.4GB RAM, 35% GPU utilization (client-side, NF-ToN-IoT)
- **Communication**: 3.8 MB/round (vs. 4.2 MB for alternative configurations)
- **Convergence**: 25-30% fewer federation rounds than alternative setups

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CLIENT SIDE                           │
│                                                               │
│  Network Flow Data → Graph Construction                      │
│         ↓                                                     │
│  ┌──────────────────────────────────────────────────┐       │
│  │         Specialized GAT Variants                  │       │
│  │  ┌─────────────┐ ┌──────────────┐ ┌────────────┐│       │
│  │  │ Temporal    │ │   Content    │ │ Behavioral ││       │
│  │  │     GAT     │ │     GAT      │ │    GAT     ││       │
│  │  │  (DDoS,DoS) │ │ (Injection,  │ │ (Backdoor, ││       │
│  │  │             │ │     XSS)     │ │  Password) ││       │
│  │  └─────────────┘ └──────────────┘ └────────────┘│       │
│  └──────────────────────────────────────────────────┘       │
│         ↓                                                     │
│  Community Detection (Louvain Algorithm)                     │
│         ↓                                                     │
│  Community Embeddings → Privacy Boundary                     │
└───────────────────────────────┬─────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ↓                       ↓
        Community Embeddings    Model Parameters
                    │                       │
                    └───────────┬───────────┘
                                ↓
┌─────────────────────────────────────────────────────────────┐
│                       SERVER SIDE                            │
│                                                               │
│  Overlay Graph Construction                                  │
│  (Communities as nodes, similarity as edges)                 │
│         ↓                                                     │
│  GraphSAGE Processing                                        │
│  (Global pattern learning via neighborhood sampling)         │
│         ↓                                                     │
│  Federated Aggregation (Performance-weighted)                │
│         ↓                                                     │
│  Updated Model Parameters → Clients                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation

### Prerequisites
```bash
Python 3.8+
CUDA 10.2+ (optional, for GPU acceleration)
```

### Dependencies
```bash
# Clone the repository
git clone https://github.com/Fouad-AlTfaily/Fed_GNN.git
cd Fed_GNN

# Install required packages
pip install -r requirements.txt
```

### Required Packages
```
torch>=1.8.0
torch-geometric>=2.0.1
numpy>=1.19.0
pandas>=1.2.0
scikit-learn>=0.24.0
networkx>=2.5
python-louvain>=0.15
matplotlib>=3.3.0
seaborn>=0.11.0
```

---

## 📁 Dataset Preparation

### Download Datasets

1. **NF-ToN-IoT Dataset**
   - Download from: https://www.kaggle.com/datasets/dhoogla/nftoniot
   - Place in: `data/nf_ton_iot/`

2. **CIC-ToN-IoT Dataset**
   - Download from: https://www.kaggle.com/datasets/dhoogla/cictoniot
   - Place in: `data/cic_ton_iot/`

### Dataset Structure
```
data/
├── nf_ton_iot/
│   ├── train.csv
│   └── test.csv
└── cic_ton_iot/
    ├── train.csv
    └── test.csv
```

---

## 💻 Usage

### Basic Training

#### Train on NF-ToN-IoT
```bash
python experiments/fedgatsage_experiment.py \
    --dataset nf_ton_iot \
    --num_clients 5 \
    --num_rounds 30 \
    --hidden_dim 256 \
    --attention_heads 8 \
    --dropout 0.2
```

#### Train on CIC-ToN-IoT
```bash
python experiments/fedgatsage_experiment.py \
    --dataset cic_ton_iot \
    --num_clients 5 \
    --num_rounds 30 \
    --hidden_dim 256 \
    --attention_heads 8 \
    --dropout 0.2
```

### Advanced Configuration

```bash
python experiments/fedgatsage_experiment.py \
    --dataset nf_ton_iot \
    --num_clients 5 \
    --num_rounds 30 \
    --hidden_dim 256 \
    --attention_heads 8 \
    --dropout 0.2 \
    --learning_rate 0.001 \
    --batch_size 512 \
    --similarity_threshold 0.7 \
    --adaptive_weighting True \
    --use_gpu True
```

### Evaluation Only

```bash
python experiments/evaluate_model.py \
    --dataset nf_ton_iot \
    --model_path checkpoints/fedgatsage_best.pth \
    --output_dir results/
```

---

## 📈 Reproducing Paper Results

### Full Experimental Pipeline

```bash
# Run complete evaluation on both datasets
python experiments/run_all_experiments.py \
    --datasets nf_ton_iot cic_ton_iot \
    --num_runs 3 \
    --save_results results/

# Generate performance comparison tables
python analysis/generate_comparison_tables.py \
    --results_dir results/ \
    --output_dir paper_results/

# Create visualization figures
python analysis/generate_figures.py \
    --results_dir results/ \
    --output_dir paper_figures/
```

### Expected Outputs
- Performance metrics (Table 1, 2, 3 from paper)
- Confusion matrices (Figure 4)
- Convergence plots (Figure 5)
- Specialized detector analysis (Figure 6)
- Community-attack correlation (Figure 7)

---

## 🔬 Key Components

### Client-Side Processing
```python
from models.gat_variants import TemporalGAT, ContentGAT, BehavioralGAT
from federated.client import FederatedClient

# Initialize specialized detectors
temporal_gat = TemporalGAT(in_dim=feature_dim, hidden_dim=256, out_dim=num_classes)
content_gat = ContentGAT(in_dim=feature_dim, hidden_dim=256, out_dim=num_classes)
behavioral_gat = BehavioralGAT(in_dim=feature_dim, hidden_dim=256, out_dim=num_classes)

# Create federated client
client = FederatedClient(
    models=[temporal_gat, content_gat, behavioral_gat],
    data=local_network_data
)

# Train and generate community embeddings
community_embeddings = client.train_and_generate_embeddings()
```

### Server-Side Processing
```python
from models.graphsage import GraphSAGE
from federated.server import FederatedServer

# Initialize server
server = FederatedServer(
    model=GraphSAGE(hidden_dim=256),
    num_clients=5
)

# Aggregate and update
global_embeddings = server.aggregate_community_embeddings(client_embeddings)
updated_params = server.update_global_model(global_embeddings)
```

---

## 📊 Project Structure

```
Fed_GNN/
├── data/                          # Dataset storage
│   ├── nf_ton_iot/
│   └── cic_ton_iot/
├── models/                        # Model architectures
│   ├── gat_variants.py           # Temporal, Content, Behavioral GAT
│   ├── graphsage.py              # Server-side GraphSAGE
│   └── ensemble.py               # Ensemble fusion
├── federated/                     # Federated learning components
│   ├── client.py                 # Client-side processing
│   ├── server.py                 # Server-side aggregation
│   └── aggregation.py            # Weighted aggregation strategies
├── preprocessing/                 # Data preprocessing
│   ├── graph_construction.py     # Network → Graph conversion
│   ├── feature_engineering.py    # Specialized feature extraction
│   └── community_detection.py    # Louvain algorithm implementation
├── experiments/                   # Experiment scripts
│   ├── fedgatsage_experiment.py  # Main training script
│   ├── evaluate_model.py         # Evaluation script
│   └── run_all_experiments.py    # Full pipeline
├── analysis/                      # Result analysis
│   ├── generate_comparison_tables.py
│   ├── generate_figures.py
│   └── statistical_tests.py
├── utils/                         # Utility functions
│   ├── metrics.py                # Evaluation metrics
│   ├── visualization.py          # Plotting functions
│   └── logger.py                 # Logging utilities
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🎓 Citation

If you use FedGATSage in your research, please cite our paper:

```bibtex
@article{altfaily2025fedgatsage,
  title={Graph-based federated learning approach for intrusion detection in IoT networks},
  author={Al Tfaily, Fouad and Ghalmane, Zakariya and Brahmia, Mohamed el Amine and Hazimeh, Hussein and Jaber, Ali and Zghal, Mourad},
  journal={Scientific Reports},
  volume={15},
  number={1},
  pages={41264},
  year={2025},
  publisher={Nature Publishing Group},
  doi={10.1038/s41598-025-25175-1}
}
```

---

## 🔍 Key Technical Contributions

### 1. Solving the Dual Limitation Problem
- **Traditional federated methods (LSTM/CNN)**: Cannot capture structural patterns due to architectural constraints
- **GNN federated methods**: Lose temporal patterns during parameter aggregation
- **FedGATSage**: Simultaneously preserves both through community abstraction and specialized detectors

### 2. Community Abstraction Mathematics

**Flow embedding creation** (combining endpoints):
```
flow_emb = [source_emb ⊕ dest_emb ⊕ (source_emb ⊗ dest_emb) ⊕ |source_emb - dest_emb|]
```

**Community aggregation** (weighted by importance):
```
community_emb = Σ(importance_score_i × node_emb_i) / Σ(importance_score_i)
```

**Similarity-based edge creation**:
```
similarity(C_i, C_j) = (C_i · C_j) / (||C_i|| · ||C_j||)
```

### 3. Complexity Analysis

- **Time Complexity (per round)**:
  - Client GAT: O(|V| · d · h + |E| · d · h)
  - Community Detection: O(|E| log |V|)
  - Server GraphSAGE: O(k · |F| · d)

- **Space Complexity**:
  - Client: O(|V| · d + L · d²)
  - Server: O(|C| · |F| · d)

- **Communication Complexity**:
  - O(|F| · d) per client → 85% reduction vs. node-level sharing

---

## 🛠️ Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size
--batch_size 256

# Or use CPU
--use_gpu False
```

**2. Community Detection Takes Too Long**
```bash
# Increase similarity threshold for faster graph construction
--similarity_threshold 0.8
```

**3. Poor Convergence**
```bash
# Increase number of rounds
--num_rounds 50

# Adjust learning rate
--learning_rate 0.0005
```

---

## 📝 Configuration Parameters

### Model Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 256 | Hidden dimension size for embeddings |
| `attention_heads` | 8 | Number of attention heads in GAT |
| `dropout` | 0.2 | Dropout rate for regularization |
| `num_layers` | 2 | Number of GNN layers |
| `learning_rate` | 0.001 | Learning rate for optimization |

### Federated Learning Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_clients` | 5 | Number of federated clients |
| `num_rounds` | 30 | Number of federation rounds |
| `local_epochs` | 3 | Local training epochs per round |
| `alpha` | 0.2 | Base weight for client contributions |
| `similarity_threshold` | 0.7 | Threshold for overlay graph edges |

---

## 🌟 Advantages Over Existing Methods

| Feature | Traditional Federated | GNN Federated | **FedGATSage** |
|---------|----------------------|---------------|----------------|
| Captures structural patterns | ❌ | ✅ | ✅ |
| Preserves temporal patterns | ❌ | ❌ | ✅ |
| Privacy preservation | ✅ | ✅ | ✅ |
| Low communication overhead | ⚠️ | ⚠️ | ✅ (85% reduction) |
| Detects coordinated attacks | ❌ | ❌ | ✅ |
| Specialized attack detection | ❌ | ❌ | ✅ |

---

## 🔮 Future Work

We are actively working on:

1. **Dynamic community detection** for evolving networks
2. **Extended attack categories** beyond current eight types
3. **Integration with differential privacy** for additional security
4. **Real-time deployment** on actual IoT infrastructure
5. **Cross-domain federated learning** across heterogeneous IoT ecosystems

---

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

This work is supported by:
- **Ektidar**: Lebanese project for youth empowerment
- **CESI LINEACT UR 7527**: Laboratory in Strasbourg, France
- **Lebanese University**: Computer Science Department, Faculty of Sciences

---

## 📧 Contact

**Fouad Al Tfaily**
- Email: fouad.altfaily@gmail.com
- GitHub: [@Fouad-AlTfaily](https://github.com/Fouad-AlTfaily)
- Paper: [Scientific Reports](https://doi.org/10.1038/s41598-025-25175-1)

---

## 📚 Related Publications

1. Al Tfaily, F., et al. (2025). "Generating realistic cyber security datasets for IoT networks with diverse complex network properties." *IoTBDS 2025*.

2. Termos, M., et al. (2024). "GDLC: A new graph deep learning framework based on centrality measures for intrusion detection in IoT networks." *Internet of Things*.

3. Ghalmane, Z., et al. (2019). "Centrality in complex networks with overlapping community structure." *Scientific Reports*.

---

<div align="center">

**⭐ If you find this work useful, please consider citing our paper and starring the repository! ⭐**

</div>
