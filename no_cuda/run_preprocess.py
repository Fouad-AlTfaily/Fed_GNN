"""
Wrapper script for preprocessing data with numpy 2.x compatibility fix.
Runs the original ../preprocess_data.py with a monkey-patch for np.array_split.

Usage (from no_cuda/ directory):
    python run_preprocess.py --input_file ../data/dummy_data.csv --output_dir ../data --num_clients 5
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Save original np.array_split
_original_array_split = np.array_split

def _patched_array_split(ary, indices_or_sections, axis=0):
    """
    Patched version of np.array_split that returns DataFrames if input is a DataFrame.
    numpy>=2.0 returns numpy arrays instead of DataFrames, breaking .to_csv() calls.
    """
    result = _original_array_split(ary, indices_or_sections, axis=axis)
    if isinstance(ary, pd.DataFrame):
        # Convert each split back to a DataFrame
        return [pd.DataFrame(r, columns=ary.columns) for r in result]
    return result

# Apply the monkey-patch
np.array_split = _patched_array_split

# Now run the original preprocessing script
if __name__ == '__main__':
    # Determine paths relative to this script (inside no_cuda/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)  # parent = Fed_GNN root
    
    # Make data dir if needed
    data_dir = os.path.join(root_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Remove this script from argv so the original script's argparse works
    sys.argv = [sys.argv[0].replace('run_preprocess.py', '../preprocess_data.py')] + sys.argv[1:]
    
    # Import the original preprocess module (from parent directory)
    import importlib.util
    original_path = os.path.join(root_dir, 'preprocess_data.py')
    spec = importlib.util.spec_from_file_location("preprocess_data", original_path)
    preprocess_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preprocess_module)
    
    # Explicitly call main() since __name__ != "__main__" when loaded as module
    preprocess_module.main()
