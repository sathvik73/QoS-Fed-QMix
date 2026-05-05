import numpy as np
import os

base_dir = r"c:\Users\gsath\OneDrive\Desktop\BTP\Current - Multi_UAV_Data_Harvesting"

runs = ["qmix run", "modelqmix run", "original run", "proposed run"]

def find_file(parent, keywords):
    for root, dirs, files in os.walk(parent):
        for f in files:
            if f.endswith('.npy') and all(key in f for key in keywords):
                return os.path.join(root, f)
    return None

EVAL_CYCLE = 50

print(f"{'Run':15} | Index Size | Actual Episodes Trained")
for run in runs:
    folder = os.path.join(base_dir, run)
    if not os.path.exists(folder): continue
    
    is_fed = find_file(folder, ["global_data"]) is not None
    f_d = find_file(folder, ["global_data.npy"]) if is_fed else find_file(folder, ["episode_data_.npy"])
    
    try:
        d_arr = np.load(f_d, allow_pickle=True)
        d = d_arr.item() if d_arr.ndim == 0 else d_arr
        size = len(d)
        print(f"{run:15} | {size:10} | {size * EVAL_CYCLE:8}")
    except Exception as e:
        print(f"Error {run}: {e}")
