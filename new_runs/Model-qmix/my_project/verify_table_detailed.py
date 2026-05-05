import numpy as np
import os

base_dir = r"c:\Users\gsath\OneDrive\Desktop\BTP\Current - Multi_UAV_Data_Harvesting"

def find_file(folder, k):
    p = os.path.join(base_dir, folder)
    for root, dirs, files in os.walk(p):
        for f in files:
            if f.endswith('.npy') and all(x in f for x in k):
                return os.path.join(root, f)
    return None

def get_arr(folder):
    is_fed = folder in ["proposed run", "original run"]
    k = ["global_data"] if is_fed else ["episode_data_"]
    f = find_file(folder, k)
    if not f: return []
    d = np.load(f, allow_pickle=True)
    return d.item() if d.ndim == 0 else d

p = get_arr("proposed run")
q = get_arr("qmix run")
o = get_arr("original run")
m = get_arr("modelqmix run")

print(f"{'Episode':10} | {'Proposed':12} | {'Original':12} | {'ModelQMIX':12} | {'Standard':12}")
print("-" * 80)

for ep in range(0, 30001, 5000):
    idx = ep // 50
    def v(arr, i):
        if not len(arr): return 0.0
        return arr[i] if i < len(arr) else arr[-1]
    
    print(f"{ep:10} | {v(p, idx):12.1f} | {v(o, idx):12.1f} | {v(m, idx):12.1f} | {v(q, idx):12.1f}")
