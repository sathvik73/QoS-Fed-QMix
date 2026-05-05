import numpy as np
import os

base_dir = r"c:\Users\gsath\OneDrive\Desktop\BTP\Current - Multi_UAV_Data_Harvesting"

runs = [
    ("Proposed", "proposed run"),
    ("Standard", "qmix run"),
    ("Model-Aided", "modelqmix run"),
    ("Original Fed", "original run")
]

def find_file(folder, k):
    p = os.path.join(base_dir, folder)
    if not os.path.exists(p): return None
    for root, dirs, files in os.walk(p):
        for f in files:
            file_name = f.lower()
            if all(key.lower() in file_name for key in k) and f.endswith(".npy"):
                return os.path.join(root, f)
    return None

print(f"{'Algorithm':20} | Total Eps | Index Ep 30000 | Value at 30k")
print("-" * 70)

for label, folder in runs:
    is_fed = "run" in folder and folder in ["proposed run", "original run"]
    file_keywords = ["global_data"] if is_fed else ["episode_data_"]
    f = find_file(folder, file_keywords)
    
    if not f:
        print(f"{label:20} | NOT FOUND ({file_keywords})")
        continue
    
    d_raw = np.load(f, allow_pickle=True)
    d = d_raw.item() if d_raw.ndim == 0 else d_raw
    size = len(d)
    
    # Check index for Episode 30,000 (30,000 / 50 = 600)
    idx = 600
    if idx < size:
        val = d[idx]
        print(f"{label:20} | {size*50:9} | {idx*50:14} | {val:12.1f}")
    else:
        print(f"{label:20} | {size*50:9} | OUT OF BOUNDS  | {d[-1]:12.1f} (Last)")
