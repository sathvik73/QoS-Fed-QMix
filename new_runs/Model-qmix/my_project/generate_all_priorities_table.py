import numpy as np
import os
import collections

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
all_p = set()
results = []
p_max_vals = collections.defaultdict(float)

# Load data at 30k episode mark
for run in runs:
    folder = os.path.join(base_dir, run["folder"])
    if not os.path.exists(folder): continue
    
    is_fed = find_file(folder, ["global_data"]) is not None
    f_p = find_file(folder, ["episode_priority_data_0.npy"]) if is_fed else find_file(folder, ["episode_priority_data_.npy"])
    f_d = find_file(folder, ["global_data.npy"]) if is_fed else find_file(folder, ["episode_data_.npy"])
    f_r = find_file(folder, ["global_rewards.npy"]) if is_fed else find_file(folder, ["episode_rewards_.npy"])
    
    try:
        p_arr = np.load(f_p, allow_pickle=True)
        d_arr = np.load(f_d, allow_pickle=True)
        r_arr = np.load(f_r, allow_pickle=True)
        
        # Unpack
        p_seq = p_arr.item() if p_arr.ndim == 0 else p_arr
        d_seq = d_arr.item() if d_arr.ndim == 0 else d_arr
        r_seq = r_arr.item() if r_arr.ndim == 0 else r_arr
        
        last_p = p_seq[-1]
        last_d = d_seq[-1]
        last_r = r_seq[-1]
        
        if isinstance(last_p, dict):
            for k, v in last_p.items():
                all_p.add(k)
                p_max_vals[k] = max(p_max_vals[k], v)
        
        results.append({
            "name": run["name"],
            "reward": last_r,
            "d": last_d,
            "p_vals": last_p if isinstance(last_p, dict) else {}
        })
    except Exception as e:
        print(f"Error logic check on {run['name']}: {e}")

sorted_p = sorted(list(all_p))

with open("comprehensive_stats_table.txt", "w") as f:
    f.write("\\begin{table*}[htbp]\n")
    f.write("\\centering\n")
    f.write("\\caption{Comprehensive Performance Benchmarking: Cumulative Rewards and All Multi-Priority Extraction Ratios at 30,000 Episodes}\n")
    f.write("\\label{tab:performance_summary_all}\n")
    
    # Generate tabular definition dynamically
    cols_def = "l" + "c" * (2 + len(sorted_p))
    f.write(f"\\begin{{tabular}}{{@{{}}{cols_def}@{{}}}}\n")
    f.write("\\toprule\n")
    
    # Headers
    headers = ["\\textbf{Algorithm}", "\\textbf{Reward}"] + [f"\\textbf{{P{p} (\%)}}" for p in sorted_p] + ["\\textbf{Total (\%)}"]
    f.write(" & ".join(headers) + " \\\\ \\midrule\n")
    
    # Rows
    for r in results:
        cells = []
        is_ours = "Proposed" in r["name"]
        
        name_cell = f"\\textbf{{{r['name']}}}" if is_ours else r["name"]
        reward_cell = f"\\textbf{{{r['reward']:.1f}}}" if is_ours else f"{r['reward']:.1f}"
        
        cells.append(name_cell)
        cells.append(reward_cell)
        
        for p in sorted_p:
            v = r["p_vals"].get(p, 0.0)
            ratio = (v / p_max_vals[p] * 100) if p_max_vals[p] > 0 else 0.0
            cells.append(f"{ratio:.1f}")
            
        total_ratio = (r["d"] / TOTAL_DATA_MAX) * 100
        cells.append(f"{total_ratio:.1f}")
        
        f.write(" & ".join(cells) + " \\\\\n")
        
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")

print(f"Comprehensive LaTeX table generated for priorities: {sorted_p}")
