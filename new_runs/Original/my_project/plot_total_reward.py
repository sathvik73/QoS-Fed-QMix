import numpy as np
import matplotlib.pyplot as plt
import os
import collections

base_dir = r"c:\Users\gsath\OneDrive\Desktop\BTP\Current - Multi_UAV_Data_Harvesting"

runs = [
    {"label": "QoS-Aware Model-Aided FedQMIX (Proposed)", "folder": "proposed_run", "c": "red", "ls": "-"},
    {"label": "Model-Aided QMIX (Centralized)", "folder": "modelqmix_run", "c": "dodgerblue", "ls": "-"},
    {"label": "Standard QMIX", "folder": "qmix_run", "c": "green", "ls": "-"},
    {"label": "Original FedQMIX (Baseline)", "folder": "original_run", "c": "darkorange", "ls": "-"}
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
    try:
        arr = np.load(filepath, allow_pickle=True)
        if arr.ndim == 0:
            return list(arr.item()) if hasattr(arr, 'item') else list(arr)
        return list(arr)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return []

# Load total reward data
for run in runs:
    folder_path = os.path.join(base_dir, run["folder"])
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        continue
    
    # Federated uses global_rewards.npy, Non-Federated uses episode_rewards_.npy
    f_reward = find_file(folder_path, ["global_rewards.npy"])
    if not f_reward:
        f_reward = find_file(folder_path, ["episode_rewards_", ".npy"])
    if not f_reward:
        f_reward = find_file(folder_path, ["incremental_test_reward.npy"])
        
    try:
        reward_data = load_arr(f_reward)
        if len(reward_data) > 0:
            data_dict[run["label"]] = {"reward": reward_data, "run_cfg": run}
        else:
            print(f"No reward data found for {run['label']} in {folder_path}")
    except Exception as e:
        print(f"Failed to load {run['label']}: {e}")

save_dir = os.path.join(base_dir, "result", "comparison_plots_detailed")
os.makedirs(save_dir, exist_ok=True)

def smooth(y, box_pts=12):
    if len(y) < box_pts: return y
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    # Prepend original values to maintain alignment
    return list(y[:box_pts-1]) + list(y_smooth)

EVAL_CYCLE = 50

fig, ax = plt.subplots(figsize=(10, 7))

for label, metrics in data_dict.items():
    if metrics["reward"]:
        y_raw = metrics["reward"]
        y_smoothed = smooth(y_raw, box_pts=15) # Slightly more smoothing for rewards
        
        run_cfg = metrics["run_cfg"]
        x_axis = []
        # All approaches simply evaluate every EVAL_CYCLE (50)
        for j in range(len(y_smoothed)):
            x_axis.append((j + 1) * EVAL_CYCLE)
        
        # Ensure x and y match lengths
        min_len = min(len(x_axis), len(y_smoothed))
        ax.plot(x_axis[:min_len], y_smoothed[:min_len], 
                label=label, color=run_cfg["c"], linestyle=run_cfg["ls"], linewidth=3.5)

ax.set_title("Total System Reward Comparison", fontsize=18, fontweight='bold')
ax.set_xlabel("Episode [log scale]", fontsize=16)
ax.set_ylabel("Total Average Reward", fontsize=16)
ax.set_xscale('log')
ax.set_xlim(40, 100000) # Creating the "short" visual distance from left edge to 10^2, ending at 10^5
# ax.set_ylim(40, 800) # Optional: based on first 5 pts showing ~50 and ~264 reward

ax.grid(True, which='major', linestyle='-')
ax.tick_params(axis='both', labelsize=14)
ax.legend(fontsize=12, loc='upper left', frameon=True)

plt.tight_layout()
output_filename = os.path.join(save_dir, "Final_Total_Reward_Log.pdf")
plt.savefig(output_filename, bbox_inches='tight')
plt.close()

print(f"Total Reward Plot Generated: {output_filename}")
