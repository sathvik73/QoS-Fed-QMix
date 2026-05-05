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

print(f"{'Episode':10} | {'Proposed Data':15} | {'Standard Data':15}")
print("-" * 50)
for idx in [0, 50, 100, 300, 500, 600]:
    pv = p[idx] if idx < len(p) else p[-1]
    qv = q[idx] if idx < len(q) else q[-1]
    print(f"{idx*50:10} | {pv:15.1f} | {qv:15.1f}")

print("\nMax Volume (200k)")
print(f"Proposed %: {p[-1]/200000.0:.2%}")
print(f"Standard %: {q[-1]/200000.0:.2%}")
