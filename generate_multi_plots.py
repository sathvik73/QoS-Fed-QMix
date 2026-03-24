import numpy as np
import matplotlib.pyplot as plt
import os
import collections

base_dir = r"c:\Users\gsath\OneDrive\Desktop\BTP\Current - Multi_UAV_Data_Harvesting"

runs = [
    {"label": "Standard QMIX", "folder": "qmix run"},
    {"label": "Model-Aided QMIX (Centralized)", "folder": "modelqmix run"},
    {"label": "Original FedQMIX (Baseline)", "folder": "original run"},
    {"label": "QoS-Aware Model-Aided FedQMIX (Proposed)", "folder": "proposed run"}
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

# 1. Load data
for run in runs:
    folder_path = os.path.join(base_dir, run["folder"])
    if not os.path.exists(folder_path):
        continue
    
    is_fed = find_file(folder_path, ["global_data"]) is not None
    
    if is_fed:
        f_data = find_file(folder_path, ["global_data.npy"])
        f_pri = find_file(folder_path, ["episode_priority_data_0.npy"]) # taking agent 0 or global? There is no global priority natively, but episode_priority_data_0 has the full array shape for FL since FL tracks globally
    else:
        f_data = find_file(folder_path, ["episode_data_.npy"])
        f_pri = find_file(folder_path, ["episode_priority_data_.npy"])
        
    try:
        data = load_arr(f_data)
        pri_data = load_arr(f_pri)
        
        data_dict[run["label"]] = {
            "data": data,
            "priority": pri_data
        }
    except Exception as e:
        print(f"Failed to load {run['label']}: {e}")

save_dir = os.path.join(base_dir, "result", "comparison_plots_detailed")
os.makedirs(save_dir, exist_ok=True)

def smooth(y, box_pts=15):
    """Moving average to smooth out RL variance."""
    if len(y) < box_pts:
         return y
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    return list(y[:box_pts-1]) + list(y_smooth)

TOTAL_DATA_LIMIT = 200000.0
EVAL_CYCLE = 50

# Pre-calculate max priority limits
priorities_found = set()
for metrics in data_dict.values():
    if "priority" in metrics and metrics["priority"]:
        for p_dict in metrics["priority"]:
            if isinstance(p_dict, dict):
                priorities_found.update(p_dict.keys())

PRIORITY_MAX = collections.defaultdict(float)
for p in priorities_found:
    max_p = 1.0 # default fallback
    for metrics in data_dict.values():
        if "priority" in metrics and metrics["priority"]:
            for p_dict in metrics["priority"]:
                 if isinstance(p_dict, dict) and p in p_dict:
                     max_p = max(max_p, p_dict[p])
    # Add a slight 2% margin just to prevent exactly 1.01 ratios if we slightly miss the absolute theoretic max 
    PRIORITY_MAX[p] = max_p + (max_p * 0.02) 

def do_plot(y_series_dict, title, ylabel, filename, x_log=False, y_limit=None):
    plt.figure(figsize=(10, 6))
    for label, y in y_series_dict.items():
        if len(y) > 0:
            y_smoothed = smooth(y, box_pts=10)
            x_axis = np.arange(1, len(y_smoothed) + 1) * EVAL_CYCLE
            plt.plot(x_axis, y_smoothed, label=label, linewidth=2.5, alpha=0.9)
            
    plt.title(title, fontsize=15, fontweight='bold')
    plt.xlabel("Total Episodes" + (" [log scale]" if x_log else ""), fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    
    if x_log:
        plt.xscale('log')
    if y_limit:
        plt.ylim(0, y_limit)
        
    plt.grid(True, which='major', linestyle='-', alpha=0.4)
    plt.grid(True, which='minor', linestyle=':', alpha=0.2)
    plt.minorticks_on()
    plt.legend(fontsize=11, loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.close()

# 2. GENERATE PLOTS
# A. Total Data Collected
total_raw = {k: v["data"] for k, v in data_dict.items()}
do_plot(total_raw, "Total Data Collected", "Data Volume (bits)", "Total_Collected_Data_Normal.png", x_log=False)
do_plot(total_raw, "Total Data Collected (Log Scale)", "Data Volume (bits)", "Total_Collected_Data_Log.png", x_log=True)

# B. Total Data Ratio
total_ratio = {k: [(d / TOTAL_DATA_LIMIT) for d in v["data"]] for k, v in data_dict.items()}
do_plot(total_ratio, "Total Data Collection Ratio", "Data Collection Ratio", "Total_Data_Ratio_Normal.png", x_log=False, y_limit=1.1)
do_plot(total_ratio, "Total Data Collection Ratio (Log Scale)", "Data Collection Ratio", "Total_Data_Ratio_Log.png", x_log=True, y_limit=1.1)

# C. Priority Data and Ratios
for p in sorted(list(priorities_found)):
    p_raw = {}
    p_ratio = {}
    
    for label, metrics in data_dict.items():
        if "priority" in metrics and metrics["priority"]:
            # Extract raw amounts
            raw_seq = [d.get(p, 0.0) if isinstance(d, dict) else 0.0 for d in metrics["priority"] ]
            p_raw[label] = raw_seq
            p_ratio[label] = [(v / PRIORITY_MAX[p]) for v in raw_seq]
            
    # Regular plots
    do_plot(p_raw, f"Priority {p} Collected Data", f"Priority {p} Volume", f"Priority_{p}_Collected_Data_Normal.png", x_log=False)
    do_plot(p_raw, f"Priority {p} Collected Data (Log Scale)", f"Priority {p} Volume", f"Priority_{p}_Collected_Data_Log.png", x_log=True)
    
    # Ratio plots
    do_plot(p_ratio, f"Priority {p} Collection Ratio", f"Priority {p} Ratio", f"Priority_{p}_Data_Ratio_Normal.png", x_log=False, y_limit=1.1)
    do_plot(p_ratio, f"Priority {p} Collection Ratio (Log Scale)", f"Priority {p} Ratio", f"Priority_{p}_Data_Ratio_Log.png", x_log=True, y_limit=1.1)

print(f"Successfully generated detailed priority and log comparison plots in {save_dir}")
