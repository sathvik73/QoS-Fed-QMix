
import numpy as np
import os
import sys

def inspect(key):
    path = f'result/qmix/RDM_priority_var/episode_{key}_.npy'
    if os.path.exists(path):
        try:
            data = np.load(path)
            print(f"--- {key} ---")
            print(f"Count: {len(data)}")
            print(f"Last 20: {data[-20:]}")
            print(f"Mean: {np.mean(data):.2f}")
            print(f"Max: {np.max(data):.2f}")
        except Exception as e:
            print(f"Error loading {key}: {e}")
    else:
        print(f"File not found: {path}")

inspect('high_priority')
inspect('rewards')
