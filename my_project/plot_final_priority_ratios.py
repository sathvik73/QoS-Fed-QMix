import numpy as np
import matplotlib.pyplot as plt
import os
import collections

base_dir = r"c:\Users\gsath\OneDrive\Desktop\BTP\Current - Multi_UAV_Data_Harvesting"

runs = [
    {"label": "QoS-Aware Model-Aided FedQMIX (Proposed)", "folder": "proposed run", "c": "red", "ls": "-", "div": 1.0},
    {"label": "Model-Aided QMIX (Centralized)", "folder": "modelqmix run", "c": "dodgerblue", "ls": "-", "div": 1.0},
    {"label": "Standard QMIX", "folder": "qmix run", "c": "green", "ls": "-", "div": 1.0},
    {"label": "Original FedQMIX (Baseline)", "folder": "original run", "c": "darkorange", "ls": "-", "div": 1.0}
]

data_dict = {}

def find_file(parent, keywords):
    for root, dirs, files in os.walk(parent):
        for f in files:
            if f.endswith('.npy'):
                match = True
                for key in keywords:
                    if key not in f:
                        match = False
                        break
                if match:
                    return os.path.join(root, f)
    return None

def load_arr(filepath):
    if not filepath: return []
    arr = np.load(filepath, allow_pickle=True)
    if arr.ndim == 0:
        return list(arr.item()) if hasattr(arr, 'item') else list(arr)
    return list(arr)

for run in runs:
    folder_path = os.path.join(base_dir, run["folder"])
    if not os.path.exists(folder_path): continue
    
    is_fed = find_file(folder_path, ["global_data"]) is not None
    if is_fed:
        f_pri = find_file(folder_path, ["episode_priority_data_0.npy"])
    else:
        f_pri = find_file(folder_path, ["episode_priority_data_.npy"])
        
    try:
        pri_data = load_arr(f_pri)
        data_dict[run["label"]] = {"priority": pri_data, "run_cfg": run}
    except Exception as e:
        print(f"Failed to load {run['label']}: {e}")

save_dir = os.path.join(base_dir, "result", "comparison_plots_detailed")
os.makedirs(save_dir, exist_ok=True)

def smooth(y, box_pts=10):
    if len(y) < box_pts: return y
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    return list(y[:box_pts-1]) + list(y_smooth)

EVAL_CYCLE = 50

priorities_found = set()
for metrics in data_dict.values():
    if metrics["priority"]:
        for p_dict in metrics["priority"]:
            if isinstance(p_dict, dict):
                priorities_found.update(p_dict.keys())

PRIORITY_MAX = collections.defaultdict(float)
for p in priorities_found:
    max_p = 1.0
    for metrics in data_dict.values():
        if metrics["priority"]:
            for p_dict in metrics["priority"]:
                 if isinstance(p_dict, dict) and p in p_dict:
                     max_p = max(max_p, p_dict[p])
    PRIORITY_MAX[p] = max_p + (max_p * 0.01)

sorted_p = sorted(list(priorities_found))
num_p = len(sorted_p)

cols = 3
rows = (num_p + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
axes = axes.flatten()

for i, p in enumerate(sorted_p):
    ax = axes[i]
    for label, metrics in data_dict.items():
        if metrics["priority"]:
            raw_seq = [d.get(p, 0.0) if isinstance(d, dict) else 0.0 for d in metrics["priority"]]
            ratio_seq = [(v / PRIORITY_MAX[p]) for v in raw_seq]
            
            y_smoothed = smooth(ratio_seq, box_pts=12)
            y_smoothed = [min(1.0, val) for val in y_smoothed]
            
            run_cfg = metrics["run_cfg"]
            div = run_cfg["div"]
            
            x_axis = []
            if "Standard" in run_cfg["label"]: # Standard QMIX
                for j in range(len(y_smoothed)):
                    if j < 1000: x_axis.append(j + 1)
                    else: x_axis.append(1000 + (j - 1000 + 1) * EVAL_CYCLE)
            else: # Model-Aided approaches
                for j in range(len(y_smoothed)):
                    x_axis.append((j + 1) * EVAL_CYCLE / div)
            
            ax.plot(x_axis, y_smoothed, label=label, color=run_cfg["c"], linestyle=run_cfg["ls"], linewidth=3.5)
            
    ax.set_title(f"Priority {p}", fontsize=18, fontweight='bold')
    ax.set_xlabel("Episode [log scale]", fontsize=16)
    ax.set_ylabel("Data collection ratio", fontsize=16)
    ax.set_xscale('log')
    ax.set_xlim(1e-1, 30000)
    ax.set_ylim(0.1, 1.0)
    
    # Matching the grid style exactly
    ax.grid(True, which='major', linestyle='-')
    ax.set_yticks(np.arange(0.1, 1.0, 0.1))
    ax.tick_params(axis='both', labelsize=14)
    
    if i == 0:
        ax.legend(fontsize=14, loc='lower right', frameon=True)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout(pad=4.0)
combined_filename = os.path.join(save_dir, "Final_Priority_Ratios_Log.pdf")
plt.savefig(combined_filename, bbox_inches='tight')
plt.close()

print(f"Final Reference-Style Plot Generated: {combined_filename}")
