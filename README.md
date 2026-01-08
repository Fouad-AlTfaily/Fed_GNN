# FedGATSage: Graph-based Federated Learning for IoT Intrusion Detection

[![Paper](https://img.shields.io/badge/Paper-Scientific%20Reports-red)](https://doi.org/10.1038/s41598-025-25175-1)
[![License](https://img.shields.io/badge/License-Open%20Access-green)](http://creativecommons.org/licenses/by-nc-nd/4.0/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8.0+-orange)](https://pytorch.org/)

Official implementation of **"Graph-based federated learning approach for intrusion detection in IoT networks"** published in *Scientific Reports* (2025).

**Authors:** Fouad Al Tfaily, Zakariya Ghalmane, Mohamed el Amine Brahmia, Hussein Hazimeh, Ali Jaber, Mourad Zghal

---

## **Abstract**

FedGATSage addresses critical limitations in federated learning for IoT intrusion detection where traditional approaches using LSTM/CNN cannot capture structural patterns, and GNN-based federated methods lose temporal patterns during parameter aggregation. Our hybrid architecture integrates client-side Graph Attention Networks (GAT) with server-side GraphSAGE through community abstraction, achieving:

- **78.58% balanced accuracy** on NF-ToN-IoT (80.24% on CIC-ToN-IoT)
- **85% communication overhead reduction**
- **Successful detection of coordinated attacks** (DDoS, DoS) that existing federated methods cannot handle
- **Privacy preservation** through community-based embeddings

---

## **Key Innovations**

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

---

## **Performance Results**

### Overall Performance

| Approach | NF-ToN-IoT ||| CIC-ToN-IoT |||
|----------|------------|------------|------|------------|------------|------|
| | Balanced Acc. | F1 | FPR/FNR | Balanced Acc. | F1 | FPR/FNR |
| **FedGATSage** | **0.785** | **0.619** | **0.066/0.223** | **0.802** | **0.800** | **0.039/0.197** |
| Centralized GAT | 0.811 | 0.797 | 0.026/0.188 | 0.840 | 0.820 | 0.024/0.169 |
| LSTM (Federated) | 0.713 | 0.610 | 0.052/0.286 | 0.759 | 0.752 | 0.043/0.225 |

**Key Achievement**: Only 2.8% performance gap compared to centralized models while maintaining complete privacy.

---

## **Architecture**

![FedGATSage Architecture](https://github.com/user-attachments/assets/b13b3206-93e4-4782-80a5-0452df76b7b6)

*Figure 1: Overview of the FedGATSage architecture showing client-side GAT processing with community detection, server-side GraphSAGE on overlay graph, and bidirectional model parameter updates.*

---

## **Project Structure**

We have organized the codebase to facilitate reproducibility and clarity:

```
Fed_GNN/
├── data/                   # Generated dataset directory (managed by preprocessing script)
├── experiments/            # Main experiment execution scripts
│   └── fedgatsage_experiment.py
├── src/                    # Core implementation modules
│   ├── federated_learning.py   # Orchestration of the federated process
│   ├── gnn_models.py           # PyTorch Geometric model definitions (GAT, GraphSAGE)
│   ├── feature_engineering.py  # Traffic feature extraction logic
│   ├── community_detection.py  # Community detection and abstraction algorithms
│   └── utils.py                # Helper functions for metrics and logging
├── preprocess_data.py      # Utility to prepare raw CSV data for federation
└── requirements.txt        # Python dependencies
```

---

## **Installation**

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

---

## **Dataset Preparation**

### Standard Datasets
You can use standard datasets like **NF-ToN-IoT** or **CIC-ToN-IoT**:
1. **NF-ToN-IoT Dataset**: [Download Link](https://www.kaggle.com/datasets/dhoogla/nftoniot)
2. **CIC-ToN-IoT Dataset**: [Download Link](https://www.kaggle.com/datasets/dhoogla/cictoniot)

### Custom Data
FedGATSage requires data to be partitioned into client-specific directories. We provide a utility script to handle this automatically.

If you provide your own CSV file, ensure it contains standard network flow columns:
*   `Src IP`, `Dst IP` (Required for graph construction)
*   `Src Port`, `Dst Port`
*   `Protocol`
*   `Flow Duration`
*   `Tot Fwd Pkts`, `Tot Bwd Pkts`
*   `Attack` (Label column)

To prepare your data:
```bash
# Process a raw CSV file into federated client datasets
python preprocess_data.py --input_file path/to/dataset.csv --output_dir data --num_clients 5
```

*If you do not have a dataset handy, the experiment script can generate synthetic "dummy" data for demonstration purposes.*

---

## **Usage**

### Running Experiments

The main entry point is `experiments/fedgatsage_experiment.py`.

**Standard Execution:**
```bash
python experiments/fedgatsage_experiment.py --data_dir data --num_clients 5 --num_rounds 15
```

**Demo Mode (Fast Verification):**
```bash
python experiments/fedgatsage_experiment.py --data_dir data --demo_mode
```

### Configuration
You can adjust parameters via command line arguments:
- `--dataset`: Choose dataset ('nf_ton_iot', 'cic_ton_iot')
- `--num_clients`: Number of federated clients
- `--num_rounds`: Number of federation rounds
- `--detector_types`: List of detectors to use (temporal, content, behavioral)
- `--device`: 'cuda' or 'cpu'

---

## **Citation**

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

## **License**

This project is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License. See [LICENSE](LICENSE) for details.

---

## **Contact**

**Fouad Al Tfaily**
- Email: fouad.altfaily@gmail.com
- GitHub: [@Fouad-AlTfaily](https://github.com/Fouad-AlTfaily)
- Paper: [Scientific Reports](https://doi.org/10.1038/s41598-025-25175-1)
