# FedGATSage — No-CUDA (CPU-Only) Setup

Run the FedGATSage experiment on machines **without an NVIDIA GPU**.

---

## Quick Start (Full Pipeline)

```bash
# 1. Create virtual environment
python3 -m venv venv --without-pip
curl -sS https://bootstrap.pypa.io/get-pip.py | venv/bin/python
source venv/bin/activate

# 2. Install PyTorch CPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. Install PyTorch Geometric
pip install torch-geometric

# 4. Install other dependencies
pip install -r requirements.txt

# 5. Generate dummy data & preprocess
python run_preprocess.py --input_file ../data/dummy_data.csv --output_dir ../data --num_clients 5

# 6. Run experiment (demo mode)
python run_experiment.py --demo_mode --data_dir ../data --device cpu --num_clients 5 --num_rounds 5 --output_dir ../results/demo
```

> **Note:** Step 1 uses `--without-pip` + bootstrap because some Linux systems (Ubuntu/Debian) don't have `python3-venv` with ensurepip. If your system has full venv support, just run `python3 -m venv venv`.

---

## Step-by-Step

### Prerequisites
- Python 3.8+
- Internet connection (for downloading packages)
- ~3 GB free disk space (for PyTorch + dependencies)

### 1. Virtual Environment
```bash
cd no_cuda/
python3 -m venv venv --without-pip
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
venv/bin/python /tmp/get-pip.py
source venv/bin/activate
```

### 2. Install PyTorch (CPU)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install PyTorch Geometric
```bash
pip install torch-geometric
```

### 4. Install Remaining Dependencies
```bash
pip install -r requirements.txt
```

### 5. Verify Installation
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); import torch_geometric; print(f'PyG {torch_geometric.__version__}'); print('OK')"
```
Expected: `CUDA: False`, all imports successful.

---

## Running Experiments

### Demo Mode (Synthetic Data — Fast)
```bash
# Generate dummy data
python run_preprocess.py --input_file ../data/dummy_data.csv --output_dir ../data --num_clients 5

# Run demo (5 rounds, 5 clients, ~2 min)
python run_experiment.py --demo_mode --data_dir ../data --device cpu --num_clients 5 --num_rounds 5 --output_dir ../results/demo
```

### With Real Dataset (Kaggle)

1. **Download** from Kaggle (both are parquet format):
   - [NF-ToN-IoT](https://www.kaggle.com/datasets/dhoogla/nftoniot) (~9 MB) — NFStream format
   - [CIC-ToN-IoT](https://www.kaggle.com/datasets/dhoogla/cictoniot) (~420 MB)

2. Place the `.parquet` files in `../data/`, then **convert + map columns**:
   ```bash
   # Convert parquet → CSV, then map column names to what the code expects
   python convert_parquet.py --all
   python map_columns.py --all
   ```
   > `map_columns.py` handles the NF-ToN-IoT NFStream→CICFlowMeter column rename, generates synthetic IPs (dataset has none), and computes missing features.

3. **Preprocess** and **run**:
   ```bash
   # For CIC-ToN-IoT:
   python run_preprocess.py --input_file ../data/CIC-ToN-IoT-V2_mapped.csv --output_dir ../data --num_clients 5
   python run_experiment.py --data_dir ../data --device cpu --num_clients 5 --num_rounds 15 --dataset cic_ton_iot --output_dir ../results/full

   # For NF-ToN-IoT:
   python run_preprocess.py --input_file ../data/NF-ToN-IoT_mapped.csv --output_dir ../data --num_clients 5
   python run_experiment.py --data_dir ../data --device cpu --num_clients 5 --num_rounds 15 --dataset nf_ton_iot --output_dir ../results/full
   ```

---

## What the Wrappers Fix

The original code was written for older library versions. The wrappers (`run_preprocess.py`, `run_experiment.py`) monkey-patch the following without modifying any original source files:

| Issue | Library | Fix |
|-------|---------|-----|
| `'int' object is not iterable` | NetworkX ≥3.0 | Convert community labels to partition format |
| `'numpy.ndarray' has no 'to_csv'` | NumPy ≥2.0 | Preserve DataFrame type in `array_split` |
| `mat1 and mat2 shapes cannot be multiplied` | (logic bug) | Apply feature engineering to test data |
| **Hang on real datasets** | Torch | Subsample GraphSAGE edges (avoid O(n²) blowup) |
| **Slow community detection** | Pandas/NetworkX | Fast groupby graph building + skip expensive vitality calc |
| **No progress feedback** | — | tqdm progress bar + ntfy.sh live notifications |

See `CHANGES.md` for technical details.

> **📱 Live notifications:** All experiment logs are sent to [ntfy.sh/asfi-fed-gnn](https://ntfy.sh/asfi-fed-gnn). Subscribe on your phone with the [ntfy app](https://ntfy.sh/) to follow progress remotely.

---

## Expected Results (Demo Mode)

With synthetic dummy data (1000 rows) and 5 training rounds:

| Metric | Value |
|--------|-------|
| Accuracy | ~70% |
| Training Loss | ~1.40 → ~1.34 |
| Time | ~2 minutes |
| Output | `../results/demo/` |

> **Note:** Low F1 scores are expected with random dummy data. For published results (78.58% balanced accuracy), use real datasets with 15+ rounds.

---

## Contributors (No-CUDA Adaptation)

- **[@asfi50](https://github.com/asfi50)** — No-CUDA compatibility wrappers and testing

---

## Files

```
no_cuda/
├── README.md           # This file
├── requirements.txt    # CPU-only Python dependencies
├── run_preprocess.py   # Data preprocessing wrapper
├── run_experiment.py   # Experiment wrapper (all fixes)
└── CHANGES.md          # Detailed changelog
```

All original source files in `../src/`, `../experiments/`, and `../preprocess_data.py` remain **unchanged**.
