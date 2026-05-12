# CPU Compatibility — Changes & Fixes

Tracks all changes made to run FedGATSage on **CPU-only machines** (no CUDA) with modern library versions.

**Date:** 2026-05-12  
**Principle:** NO original source files were modified. All fixes are in wrapper scripts.

---

## Environment

| Component | Version |
|-----------|---------|
| Python | 3.12.3 |
| PyTorch (CPU) | 2.11.0+cpu |
| PyG (torch-geometric) | 2.7.0 |
| NumPy | 2.4.3 |
| NetworkX | 3.6.1 |
| Pandas | 3.0.3 |

---

## Files in `no_cuda/`

| File | Purpose |
|------|---------|
| `README.md` | Full setup and usage instructions |
| `requirements.txt` | CPU-only Python dependencies |
| `run_preprocess.py` | Wrapper for `../preprocess_data.py` |
| `run_experiment.py` | Main wrapper with all compatibility fixes |
| `CHANGES.md` | This file — documents all changes |

---

## Bugs Fixed (via Monkey-Patching)

### Fix 1: NetworkX 3.x `modularity()` API Change
**File affected (not modified):** `src/community_detection.py:compute_modularity_vitality()`  
**Problem:** The code passes `communities.values()` (a list of integer community labels) to `nx.community.modularity()`. NetworkX 3.x requires a **partition** (list of sets of nodes).  
**Error:** `'int' object is not iterable`  
**Fix:** Monkey-patched to convert the community dict `{node: community_id}` into a proper partition `[{nodes_in_comm1}, {nodes_in_comm2}, ...]` before calling `nx.community.modularity()`.

### Fix 2: NumPy 2.x `np.array_split` DataFrame Handling
**File affected (not modified):** `preprocess_data.py:save_split_data()`  
**Problem:** `np.array_split(df, n)` returns numpy arrays in NumPy 2.x instead of DataFrames.  
**Error:** `'numpy.ndarray' object has no attribute 'to_csv'`  
**Fix:** Monkey-patched `np.array_split` to detect DataFrame input and convert output back to DataFrames.

### Fix 3: Missing Feature Engineering in Evaluation
**File affected (not modified):** `experiments/fedgatsage_experiment.py:evaluate_system()`  
**Problem:** The evaluation function calls `_process_to_graph()` directly on raw test CSV, skipping the feature engineering pipeline that was applied during training. This causes a dimension mismatch between model weights and test data.  
**Error:** `mat1 and mat2 shapes cannot be multiplied (296x14 and 17x256)`  
**Fix:** Monkey-patched `evaluate_system()` to apply the same feature engineering pipeline to test data before processing.

---

## Experiment Results (Demo Mode — Dummy Data)

| Metric | Value |
|--------|-------|
| Accuracy | 70.5% |
| Balanced Accuracy | 25.0% |
| Macro F1 | 0.207 |
| Weighted F1 | 0.583 |
| Training Loss | 1.407 → 1.343 |
| Total Time | ~96 seconds |
| Rounds | 5 |
| Clients | 5 |
| Detectors | temporal, content, behavioral |

**Note:** Low F1 scores are expected due to:
1. Randomly generated dummy data (not real network traffic)
2. Only 5 training rounds (vs 15 recommended)
3. Small dataset (160 records/client)

For published results (78.58% balanced accuracy), use real datasets with 15+ rounds.

---

## Original Files (UNCHANGED)

- `src/gnn_models.py`
- `src/federated_learning.py`
- `src/feature_engineering.py`
- `src/community_detection.py`
- `src/utils.py`
- `experiments/fedgatsage_experiment.py`
- `preprocess_data.py`
- `requirements.txt`
