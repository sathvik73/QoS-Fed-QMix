import numpy as np
import matplotlib.pyplot as plt
import os
import collections
import json

base_dir = r"c:\Users\gsath\OneDrive\Desktop\BTP\Current - Multi_UAV_Data_Harvesting"

runs = [
    {"label": "QoS-Aware Model-Aided FedQMIX (Proposed)", "folder": "proposed_run", "tag": "RDM_proposedf", "c": "red", "ls": "-", "is_fed": True},
    {"label": "Model-Aided QMIX (Centralized)", "folder": "modelqmix_run", "tag": "RDM_model_qmix", "c": "dodgerblue", "ls": "--", "is_fed": False},
    {"label": "Standard QMIX", "folder": "qmix_run", "tag": "RDM_qmix", "c": "green", "ls": "-.", "is_fed": False},
    {"label": "Original FedQMIX (Baseline)", "folder": "original_run", "tag": "RDM_original", "c": "darkorange", "ls": ":", "is_fed": True}
]

def smooth(y, box_pts=12):
    if len(y) < box_pts: return y
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    return list(y[:box_pts-1]) + list(y_smooth)

def get_x_axis(data_len, run_cfg):
    evaluate_cycle = run_cfg.get("evaluate_cycle", 50)
    # Standard evaluation cycle evaluates based on configuration. We do not compress any data,
    # mapping index i to i * evaluate_cycle allows plots to truncate naturally using xlim.
    return [i*evaluate_cycle for i in range(data_len)]

# Find the valid path to the result folder for a run
def find_result_folder(run):
    # First, check if the unified pipeline output exists locally
    unified_dir = os.path.join(base_dir, "result", "qmix", run.get("tag", ""))
    if os.path.exists(unified_dir):
        return unified_dir
        
    # Fallback to scattered legacy folders
    search_dir = os.path.join(base_dir, run["folder"], "result", "qmix")
    if not os.path.exists(search_dir):
        return None
    for subdir in os.listdir(search_dir):
        full_subdir = os.path.join(search_dir, subdir)
        if os.path.isdir(full_subdir) and subdir.startswith("RDM_"):
            return full_subdir
    return None

data_dict = {}

for run in runs:
    sub_dir = find_result_folder(run)
    if not sub_dir:
        print(f"Result folder not found for {run['label']}")
        continue
    
    is_fed = run["is_fed"]
    f_total = os.path.join(sub_dir, "global_data.npy") if is_fed else os.path.join(sub_dir, "episode_data_.npy")
    f_rew = os.path.join(sub_dir, "global_rewards.npy") if is_fed else os.path.join(sub_dir, "episode_rewards_.npy")
    f_pri = os.path.join(sub_dir, "episode_priority_data_0.npy") if is_fed else os.path.join(sub_dir, "episode_priority_data_.npy")
    f_params = os.path.join(sub_dir, "log_params.txt")
    
    evaluate_cycle = 50
    if os.path.exists(f_params):
        try:
            with open(f_params, 'r') as f:
                params = json.load(f)
                evaluate_cycle = params.get("evaluate_cycle", 50)
        except:
            pass
    run["evaluate_cycle"] = evaluate_cycle
    
    # Truncate to maximum 601 data points to effectively cutoff at 30k episodes mathematically.
    # length 601 corresponds to exactly 0 to 30,000 (at evaluate_cycle=50)
    # length 600 corresponds to 50 to 30,000
    
    total_data = list(np.load(f_total, allow_pickle=True)) if os.path.exists(f_total) else []
    reward_data = list(np.load(f_rew, allow_pickle=True)) if os.path.exists(f_rew) else []
    priority_data = list(np.load(f_pri, allow_pickle=True)) if os.path.exists(f_pri) else []
    
    # Do absolutely NO structural modification/truncation or prepending to raw dataset.
    # Preserve exact lengths (601 for proposed/original, 600 for modelqmix, 1580 for std qmix).
    
    metrics = {
        "total": total_data,
        "reward": reward_data,
        "priority": priority_data,
        "run_cfg": run
    }
    
    if len(metrics["total"]) > 0:
        data_dict[run["label"]] = metrics

save_dir = os.path.join(base_dir, "result", "final_paper_plots")
os.makedirs(save_dir, exist_ok=True)

# Determine maximums for ratio calculations
MAX_TOTAL = 0
MAX_REWARD = 0
PRIORITY_MAX = collections.defaultdict(float)
priorities_found = set()

for lbl, metrics in data_dict.items():
    if len(metrics["total"]) > 0:
        MAX_TOTAL = max(MAX_TOTAL, np.max(metrics["total"]))
    if len(metrics["reward"]) > 0:
        MAX_REWARD = max(MAX_REWARD, np.max(metrics["reward"]))
        
    if len(metrics["priority"]) > 0:
        for p_dict in metrics["priority"]:
            if isinstance(p_dict, dict):
                priorities_found.update(p_dict.keys())

for p in priorities_found:
    max_p = 1.0
    for metrics in data_dict.values():
        if len(metrics["priority"]) > 0:
            for p_dict in metrics["priority"]:
                 if isinstance(p_dict, dict) and p in p_dict:
                     max_p = max(max_p, p_dict[p])
    PRIORITY_MAX[p] = max_p + (max_p * 0.01)

MAX_TOTAL = MAX_TOTAL if MAX_TOTAL > 0 else 1.0
MAX_REWARD = MAX_REWARD if MAX_REWARD > 0 else 1.0

def plot_and_save(filename, y_data_key, y_label, title, scales=["linear", "log"], is_priority=None):
    for scale in scales:
        plt.figure(figsize=(10, 6))
        
        has_data = False
        for label, metrics in data_dict.items():
            if is_priority is not None:
                if len(metrics["priority"]) > 0:
                    raw_seq = [d.get(is_priority, 0.0) if isinstance(d, dict) else 0.0 for d in metrics["priority"]]
                    ratio_seq = [(v / PRIORITY_MAX[is_priority]) for v in raw_seq]
                    y_smoothed = smooth(ratio_seq, box_pts=12)
                    y_smoothed = [min(1.0, val) for val in y_smoothed]
                    x_axis = get_x_axis(len(metrics["priority"]), metrics["run_cfg"])
                    
                    # align length
                    min_len = min(len(x_axis), len(y_smoothed))
                    x_plot = x_axis[:min_len]
                    y_plot = y_smoothed[:min_len]
                    
                    # Filter out evaluation at episode 0 so we don't connect to y-axis
                    x_plot_f = [x for x in x_plot if x > 0]
                    y_plot_f = [y for x, y in zip(x_plot, y_plot) if x > 0]
                    
                    plt.plot(x_plot_f, y_plot_f, label=label, color=metrics["run_cfg"]["c"], linestyle=metrics["run_cfg"]["ls"], linewidth=4.0)
                    has_data = True
            else:
                d = metrics.get(y_data_key, [])
                if len(d) > 0:
                    # Smoothing total data and rewards is also good for visualization
                    y = np.array(d)
                    if y_data_key == "total":
                        y = y / MAX_TOTAL
                    
                    y_smoothed = smooth(y.tolist(), box_pts=12)
                    x_axis = get_x_axis(len(d), metrics["run_cfg"])
                    
                    # align length
                    min_len = min(len(x_axis), len(y_smoothed))
                    x_plot = x_axis[:min_len]
                    y_plot = y_smoothed[:min_len]
                    
                    # Filter out evaluation at episode 0 so we don't connect to y-axis
                    x_plot_f = [x for x in x_plot if x > 0]
                    y_plot_f = [y for x, y in zip(x_plot, y_plot) if x > 0]
                    
                    plt.plot(x_plot_f, y_plot_f, label=label, color=metrics["run_cfg"]["c"], linestyle=metrics["run_cfg"]["ls"], linewidth=4.0)
                    has_data = True
                    
        if has_data:
            plt.title(f"{title} ({scale.capitalize()} Scale)", fontsize=22)
            plt.xlabel("Episodes", fontsize=20)
            plt.ylabel(y_label, fontsize=20)
            
            plt.xscale(scale)
            if scale == 'log':
                plt.xlim(10, 30000)
            else:
                plt.xlim(0, 30000)
                
            plt.grid(True, which='major', linestyle='-', alpha=0.7)
            if scale == 'log':
                plt.grid(True, which='minor', linestyle=':', alpha=0.5)
            
            plt.tick_params(axis='both', labelsize=18)
            plt.legend(fontsize=16, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True)
            plt.tight_layout()
            
            save_path_png = os.path.join(save_dir, f"{filename}_{scale}.png")
            save_path_pdf = os.path.join(save_dir, f"{filename}_{scale}.pdf")
            plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
            plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight')
            print(f"Saved: {save_path_png} and .pdf")
            
        plt.close()

# 1. Total Data Collection
plot_and_save("Total_Data_Ratio", "total", "Total Data Collection Ratio", "Overall Data Collection")

# 2. Total Rewards
plot_and_save("Total_Reward", "reward", "Total Reward", "Overall Episodic Reward")

# 3. Individual Priorities
sorted_p = sorted(list(priorities_found))
for p in sorted_p:
    plot_and_save(f"Priority_{p}_Ratio", "priority", f"Priority {p} Ratio", f"Data Collection Ratio for Priority {p}", is_priority=p)

print("\nAll individual plots saved!")

# Now also plot the combined figure for priorities to match the exact aesthetic of the prior run
cols = 3
num_p = len(sorted_p)
rows = (num_p + cols - 1) // cols

for scale in ["linear", "log"]:
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten()

    for i, p in enumerate(sorted_p):
        ax = axes[i]
        for label, metrics in data_dict.items():
            if len(metrics["priority"]) > 0:
                raw_seq = [d.get(p, 0.0) if isinstance(d, dict) else 0.0 for d in metrics["priority"]]
                ratio_seq = [(v / PRIORITY_MAX[p]) for v in raw_seq]
                
                y_smoothed = smooth(ratio_seq, box_pts=12)
                y_smoothed = [min(1.0, val) for val in y_smoothed]
                
                run_cfg = metrics["run_cfg"]
                x_axis = get_x_axis(len(metrics["priority"]), run_cfg)
                
                min_len = min(len(x_axis), len(y_smoothed))
                x_plot = x_axis[:min_len]
                y_plot = y_smoothed[:min_len]
                
                # Filter out evaluation at episode 0 so we don't connect to y-axis
                x_plot_f = [x for x in x_plot if x > 0]
                y_plot_f = [y for x, y in zip(x_plot, y_plot) if x > 0]
                
                ax.plot(x_plot_f, y_plot_f, label=label, color=run_cfg["c"], linestyle=run_cfg["ls"], linewidth=4.0)
                
        ax.set_title(f"Priority {p}", fontsize=22)
        ax.set_xlabel("Episode" + (" [log scale]" if scale == "log" else ""), fontsize=20)
        ax.set_ylabel("Data collection ratio", fontsize=20)
        
        ax.set_xscale(scale)
        if scale == "log":
            ax.set_xlim(10, 30000)
        else:
            ax.set_xlim(0, 30000)
            
        ax.set_ylim(0.0, 1.05)
        
        ax.grid(True, which='major', linestyle='-', alpha=0.7)
        ax.tick_params(axis='both', labelsize=18)

    for j in range(len(sorted_p), len(axes)):
        fig.delaxes(axes[j])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=18, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=True)
    plt.tight_layout(pad=4.0, rect=[0, 0.08, 1, 1])
    combined_filename_png = os.path.join(save_dir, f"Combined_Priority_Ratios_{scale.capitalize()}.png")
    combined_filename_pdf = os.path.join(save_dir, f"Combined_Priority_Ratios_{scale.capitalize()}.pdf")
    plt.savefig(combined_filename_png, bbox_inches='tight', dpi=300)
    plt.savefig(combined_filename_pdf, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {combined_filename_png} and .pdf")

print(f"Done! Find results in {save_dir}")
