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

### With Real Dataset
```bash
# Download NF-ToN-IoT or CIC-ToN-IoT dataset first, then:
python run_preprocess.py --input_file /path/to/dataset.csv --output_dir ../data --num_clients 5

# Full experiment (15 rounds as per paper)
python run_experiment.py --data_dir ../data --device cpu --num_clients 5 --num_rounds 15 --dataset cic_ton_iot --output_dir ../results/full
```

---

## What the Wrappers Fix

The original code was written for older library versions. The wrappers (`run_preprocess.py`, `run_experiment.py`) monkey-patch the following without modifying any original source files:

| Issue | Library | Fix |
|-------|---------|-----|
| `'int' object is not iterable` | NetworkX ≥3.0 | Convert community labels to partition format |
| `'numpy.ndarray' has no 'to_csv'` | NumPy ≥2.0 | Preserve DataFrame type in `array_split` |
| `mat1 and mat2 shapes cannot be multiplied` | (logic bug) | Apply feature engineering to test data |

See `CHANGES.md` for technical details.

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
