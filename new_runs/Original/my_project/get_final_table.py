import numpy as np
import os

base_dir = r"c:\Users\gsath\OneDrive\Desktop\BTP\Current - Multi_UAV_Data_Harvesting"

runs = [
    {"name": "Proposed FedQMIX (QoS-Aware)", "folder": "proposed run"},
    {"name": "Original FedQMIX (Baseline)", "folder": "original run"},
    {"name": "Model-Aided QMIX (Centralized)", "folder": "modelqmix run"},
    {"name": "Standard QMIX (Model-Free)", "folder": "qmix run"}
]

def find_file(parent, keywords):
    for root, dirs, files in os.walk(parent):
        for f in files:
            if f.endswith('.npy') and all(k in f for k in keywords):
                return os.path.join(root, f)
    return None

TOTAL_DATA_MAX = 200000.0
P_MAX = {1: 19430.2, 2: 12000.0, 3: 40000.0}

results = []

for run in runs:
    folder = os.path.join(base_dir, run["folder"])
    if not os.path.exists(folder): continue
    
    is_fed = find_file(folder, ["global_data"]) is not None
    f_d = find_file(folder, ["global_data.npy"]) if is_fed else find_file(folder, ["episode_data_.npy"])
    f_r = find_file(folder, ["global_rewards.npy"]) if is_fed else find_file(folder, ["episode_rewards_.npy"])
    f_p = find_file(folder, ["episode_priority_data_0.npy"]) if is_fed else find_file(folder, ["episode_priority_data_.npy"])
    
    try:
        d_arr = np.load(f_d, allow_pickle=True)
        r_arr = np.load(f_r, allow_pickle=True)
        p_arr = np.load(f_p, allow_pickle=True)
        
        d = d_arr.item() if d_arr.ndim == 0 else d_arr
        r = r_arr.item() if r_arr.ndim == 0 else r_arr
        p = p_arr.item() if p_arr.ndim == 0 else p_arr
        
        last_d = d[-1]
        last_r = r[-1]
        last_p = p[-1]
        
        p1 = last_p.get(1, 0) if isinstance(last_p, dict) else 0
        p2 = last_p.get(2, 0) if isinstance(last_p, dict) else 0
        p3 = last_p.get(3, 0) if isinstance(last_p, dict) else 0
        
        P_MAX[1] = max(P_MAX[1], p1)
        P_MAX[2] = max(P_MAX[2], p2)
        P_MAX[3] = max(P_MAX[3], p3)
        
        results.append({
            "name": run["name"],
            "reward": last_r,
            "d": last_d,
            "p1": p1,
            "p2": p2,
            "p3": p3
        })
    except Exception as e:
        print(f"Error logic check on {run['name']}: {e}")

print("\nFinal Result Table at 30k Episodes:")
print("-" * 100)
with open("final_table_metrics.txt", "w") as f:
    f.write(f"{'Algorithm':40} | Reward  | P3 Ratio | P2 Ratio | P1 Ratio | Total Ratio\n")
    f.write("-" * 100 + "\n")
    for r in results:
        p1_r = (r['p1'] / P_MAX[1]) * 100
        p2_r = (r['p2'] / P_MAX[2]) * 100
        p3_r = (r['p3'] / P_MAX[3]) * 100
        tot_r = (r['d'] / TOTAL_DATA_MAX) * 100
        f.write(f"{r['name']:40} | {r['reward']:7.1f} | {p3_r:8.1f}% | {p2_r:8.1f}% | {p1_r:8.1f}% | {tot_r:10.1f}%\n")
